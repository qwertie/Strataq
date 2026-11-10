// src/common/experts/ram_tier.cpp - the RAM tier's arithmetic and the host-memory readings it is sized by.  Read
// the header first.
#include "strata/common/experts/ram_tier.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>   // host_statistics64: the available memory
#else
#include <unistd.h>
#endif

namespace strata::core {

namespace detail {

bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes) {
    bytes = 0;
    if (!stat.valid) return false;

    // memory.stat's counters can race memory.current, so bound them to charged usage first.  The whole clean page
    // cache is reclaimable (149: MemAvailable counts active file pages too); the kernel keeps a low-watermark share.
    uint64_t reclaimable = std::min(std::max(stat.file, stat.inactive_file), stat.current);
    reclaimable = stat.file_dirty >= reclaimable ? 0 : reclaimable - stat.file_dirty;
    reclaimable = stat.file_writeback >= reclaimable ? 0 : reclaimable - stat.file_writeback;
    reclaimable -= std::min(reclaimable / 2, limit / 64);

    // Reclaiming clean file pages reduces usage; saturating subtraction also handles a transient over-limit read.
    const uint64_t usage_after_reclaim = stat.current - reclaimable;
    bytes = usage_after_reclaim < limit ? limit - usage_after_reclaim : 0;
    return true;
}

uint64_t clamp_resident_budget(uint64_t requested, uint64_t physical, uint64_t commit, uint64_t headroom) {
    const uint64_t available = std::min(physical, commit);
    const uint64_t room = available > headroom ? available - headroom : 0;
    if (requested <= room) return requested;
    const uint64_t margin = 256ull << 20;
    return room > margin ? room - margin : 0;
}

bool pin_depends_on_reclaim(uint64_t free_now, uint64_t bytes, uint64_t reserve) {
    return bytes > 0 && free_now < bytes + reserve;
}

bool wait_for_ram(uint64_t need, double max_wait_s, double step_s, double note_s,
                  const std::function<bool(uint64_t&)>& sample, const std::function<void(double)>& sleep,
                  const std::function<void(double, uint64_t)>& progress, double& waited, uint64_t& available) {
    waited = 0.0;
    if (!sample(available)) return false;
    double next_note = 0.0;
    while (available < need && waited < max_wait_s) {
        if (progress && waited >= next_note) {
            progress(waited, available);
            next_note = waited + note_s;
        }
        sleep(step_s);
        waited += step_s;
        if (!sample(available)) return false;
    }
    return available >= need;
}

bool pin_step_fits(uint64_t available, uint64_t step, uint64_t reserve) {
    return available >= step + reserve;
}

uint64_t meminfo_bytes(const std::string& meminfo_text, const std::string& key) {
    std::istringstream in(meminfo_text);
    std::string line;
    const std::string want = key + ":";
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string k, unit;
        uint64_t value = 0;
        if (fields >> k >> value && k == want) {
            fields >> unit;
            return unit == "kB" ? value * 1024 : value;
        }
    }
    return 0;
}

uint64_t numa_pin_free(uint64_t global_free, const std::vector<uint64_t>& node_free) {
    if (node_free.size() < 2) return global_free;
    const uint64_t smallest = *std::min_element(node_free.begin(), node_free.end());
    return std::min<uint64_t>(global_free, smallest * (uint64_t) node_free.size());
}

std::array<std::pair<uint64_t, uint64_t>, 3> numa_slice_ranges(uint64_t slot_off, const uint64_t role_bytes[3],
                                                               int slice, uint64_t page) {
    assert(slice == 0 || slice == 1);
    std::array<std::pair<uint64_t, uint64_t>, 3> out{};
    uint64_t start = slot_off;
    for (int r = 0; r < 3; ++r) {
        const uint64_t mid = start + role_bytes[r] / 2, end = start + role_bytes[r];
        if (slice == 0) {
            out[r] = {start / page * page, (mid + page - 1) / page * page};
        } else {
            const uint64_t lo = (mid + page - 1) / page * page, hi = end / page * page;
            out[r] = hi > lo ? std::make_pair(lo, hi) : std::make_pair(lo, lo);
        }
        start = end;
    }
    return out;
}

std::string page_cache_hint(const std::string& meminfo_text) {
    const uint64_t cache = meminfo_bytes(meminfo_text, "Cached") + meminfo_bytes(meminfo_text, "Buffers");
    if (cache < (1ull << 30)) return "";
    char msg[420];
    std::snprintf(msg, sizeof msg,
                  "; %.1f GiB of the RAM is file cache (earlier runs' model files, which Linux gives back on demand - "
                  "'free' showing little is normal). If the start still fails, free it with: "
                  "sync; echo 1 | sudo tee /proc/sys/vm/drop_caches   (or start strata serve with "
                  "STRATA_DROP_CACHE_ON_EXIT=1 so it releases its model files when it stops)",
                  (double) cache / 1073741824.0);
    return msg;
}

bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err) {
    offsets.clear();
    bytes = 0;
    err.clear();
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers) {
        err = "FileExpertSource: invalid geometry for the cache complement plan";
        return false;
    }
    if ((uint64_t) n_layers > (uint64_t) std::numeric_limits<size_t>::max() / (uint64_t) n_expert) {
        err = "FileExpertSource: cache complement index table is too large";
        return false;
    }
    const size_t count = (size_t) n_layers * (size_t) n_expert;
    std::vector<uint8_t> omitted(count, 0);
    auto mark_pairs = [&](const std::vector<std::pair<int32_t, int32_t>>& pairs, uint8_t bit,
                          const char* label) -> bool {
        for (const auto& pair : pairs) {
            if (pair.first < 0 || pair.second < 0 || pair.first >= n_layers || pair.second >= n_expert) {
                err = std::string("FileExpertSource: ") + label + " pair is outside the expert geometry";
                return false;
            }
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if ((omitted[index] & bit) != 0) {
                err = std::string("FileExpertSource: duplicate ") + label + " pair in the cache complement plan";
                return false;
            }
            if (bit == 2 && (omitted[index] & 1) != 0) {
                err = "FileExpertSource: the primary and additional GPU expert tiers overlap";
                return false;
            }
            omitted[index] |= bit;
        }
        return true;
    };
    if (!mark_pairs(primary_gpu_pairs, 1, "primary GPU") ||
        !mark_pairs(additional_gpu_pairs, 2, "additional GPU")) return false;
    for (uint64_t blob_bytes : layer_blob_bytes) {
        if (blob_bytes == 0) {
            err = "FileExpertSource: cache complement layer has zero-sized expert blobs";
            return false;
        }
    }

    offsets.assign(count, kNoCacheComplement);
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        const uint64_t blob_bytes = layer_blob_bytes[(size_t) layer];
        for (int64_t expert = 0; expert < n_expert; ++expert) {
            const size_t index = (size_t) layer * (size_t) n_expert + (size_t) expert;
            if (omitted[index] != 0) continue;
            if (bytes > std::numeric_limits<uint64_t>::max() - blob_bytes) {
                offsets.clear();
                bytes = 0;
                err = "FileExpertSource: cache complement size overflows";
                return false;
            }
            offsets[index] = bytes;
            bytes += blob_bytes;
        }
    }
    if (bytes > (uint64_t) std::numeric_limits<size_t>::max()) {
        offsets.clear();
        bytes = 0;
        err = "FileExpertSource: cache complement exceeds the host address space";
        return false;
    }
    return true;
}

const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback) {
    if (complement_host != nullptr && index < offsets.size() && offsets[index] != kNoCacheComplement)
        return complement_host + (size_t) offsets[index];
    return mapped_fallback;
}

int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from) {
    if (base_bytes > budget) return -1;
    const int64_t slots = (int64_t) slot_bytes.size();
    if (lend_from < 0 || lend_from > slots) lend_from = slots;   // no lend region: only the experts no slot holds
    int64_t keep = slots;
    uint64_t bytes = base_bytes;
    while (keep > lend_from) {
        const uint64_t b = slot_bytes[(size_t) keep - 1];
        if (b > budget - bytes) break;
        bytes += b;
        --keep;
    }
    return keep;
}

int64_t append_stage_lend_regions(int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
                                  std::vector<uint64_t>& offsets, uint64_t& bytes, uint64_t cap_bytes,
                                  const std::vector<std::vector<std::pair<int32_t, int32_t>>>& regions,
                                  std::string& err) {
    const size_t blobs = (size_t) (n_layers * n_expert);
    if (n_layers <= 0 || n_expert <= 0 || layer_blob_bytes.size() != (size_t) n_layers || offsets.size() != blobs) {
        err = "append_stage_lend_regions: the plan does not match the geometry";
        return -1;
    }
    std::vector<std::pair<int32_t, int32_t>> sorted;
    for (const auto& region : regions) {   // validate the whole request before touching the plan
        for (const auto& pair : region)
            if (pair.first < 0 || pair.first >= n_layers || pair.second < 0 || pair.second >= n_expert) {
                err = "append_stage_lend_regions: a lend-region expert is outside the model";
                return -1;
            }
        sorted = region;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i] == sorted[i - 1]) {
                err = "append_stage_lend_regions: a stage's lend region repeats an expert";
                return -1;
            }
    }
    int64_t kept = 0;
    for (const auto& region : regions) {
        for (const auto& pair : region) {
            const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
            if (offsets[index] != kNoCacheComplement) continue;   // already in the copy: the complement, or a
                                                                  // CUDA0 lend slot the region is asked to skip
            const uint64_t b = layer_blob_bytes[(size_t) pair.first];
            if (b > cap_bytes - bytes) break;                     // this stage's rest keeps the file fallback
            offsets[index] = bytes;
            bytes += b;
            ++kept;
        }
    }
    return kept;
}

bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out) {
    if (in == out || in >= offsets.size() || out >= offsets.size() || offsets[in] == kNoCacheComplement ||
        offsets[out] != kNoCacheComplement) return false;
    offsets[out] = offsets[in];
    offsets[in] = kNoCacheComplement;
    return true;
}

}  // namespace detail

namespace {

#if defined(__linux__)
bool read_cgroup_memory_stat(const std::filesystem::path& path, uint64_t current,
                             detail::CgroupMemoryStat& stat) {
    std::ifstream input(path / "memory.stat");
    if (!input) return false;

    bool inactive_file = false, file_dirty = false, file_writeback = false, file = false;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        if (!(fields >> key >> value)) return false;
        fields >> std::ws;
        if (!fields.eof()) return false;

        if (key == "inactive_file") {
            if (inactive_file) return false;
            inactive_file = true;
            stat.inactive_file = value;
        } else if (key == "file_dirty") {
            if (file_dirty) return false;
            file_dirty = true;
            stat.file_dirty = value;
        } else if (key == "file_writeback") {
            if (file_writeback) return false;
            file_writeback = true;
            stat.file_writeback = value;
        } else if (key == "file") {   // 149: optional (every v2 kernel has it); 0 = inactive_file alone
            if (file) return false;
            file = true;
            stat.file = value;
        }
    }
    if (!input.eof() || !inactive_file || !file_dirty || !file_writeback) return false;
    stat.current = current;
    stat.valid = true;
    return true;
}
#endif

}  // namespace

namespace detail {

bool host_available_memory(HostMemory& m, const std::string& meminfo, const std::string& self_cgroup,
                           const std::string& cgroup_root) {
    m = HostMemory{};
#if defined(_WIN32)
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return false;
    m.available = (uint64_t) status.ullAvailPhys;
    m.commit = (uint64_t) status.ullAvailPageFile;
    m.commit_limit = (uint64_t) status.ullTotalPageFile;
    return m.available > 0;
#elif defined(__linux__)
    // MemAvailable includes reclaimable page cache, unlike _SC_AVPHYS_PAGES.
    std::ifstream info(meminfo);
    std::string line;
    uint64_t bytes = 0;
    while (std::getline(info, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == "MemAvailable:" && unit == "kB" &&
            value <= std::numeric_limits<uint64_t>::max() / 1024) bytes = value * 1024;
    }
    if (bytes == 0) return false;
    // Account for the tightest cgroup ancestor limit when its normal mount is visible.
    // This is a point-in-time guard, not a reservation against concurrent allocations.
    std::ifstream groups(self_cgroup);
    if (!groups) return false;
    const std::filesystem::path root(cgroup_root);
    bool v2 = false;
    std::string v1_path;   // the memory controller's group (cgroup v1), when there is no v2 line
    while (std::getline(groups, line)) {
        if (line.rfind("0::/", 0) != 0) {
            // v1: "N:controller[,controller]:/path"
            const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
            if (c2 == std::string::npos) continue;
            std::istringstream ctl(line.substr(c1 + 1, c2 - c1 - 1));
            for (std::string c; std::getline(ctl, c, ',');)
                if (c == "memory") v1_path = line.substr(c2 + 1);
            continue;
        }
        auto path = (root / line.substr(4)).lexically_normal();
        if (path.string().rfind(root.string(), 0) != 0 || !std::filesystem::is_directory(path)) return false;
        v2 = true;
        while (path.string().rfind(root.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.max"), current_file(path / "memory.current");
            std::string limit;
            uint64_t current = 0;
            const bool readable = bool(limit_file >> limit) && bool(current_file >> current);
            // The host's root cgroup has no memory.max; ordinary child groups must expose their limits.
            if (!readable && !(path == root && !std::filesystem::exists(path / "memory.max") &&
                               std::filesystem::exists(path / "cgroup.controllers"))) return false;
            if (readable && limit != "max") {
                try {
                    size_t consumed = 0;
                    const uint64_t cap = std::stoull(limit, &consumed);
                    if (consumed != limit.size()) return false;
                    CgroupMemoryStat stat;
                    if (!read_cgroup_memory_stat(path, current, stat)) return false;
                    uint64_t cgroup_available = 0;
                    if (!cgroup_available_bytes(cap, stat, cgroup_available)) return false;
                    bytes = std::min(bytes, cgroup_available);
                    m.cgroup_limit = std::min(m.cgroup_limit, cap);
                } catch (...) { return false; }
            }
            if (path == root) break;
            path = path.parent_path();
        }
    }
    if (!v2 && !v1_path.empty()) {
        // #633: cgroup v1 (older Docker hosts, RHEL 7/8): the memory controller's group and its ancestors.  An
        // unlimited group says a number near 2^63 (rounded to its page size); a group whose files are not
        // visible (no mount in this namespace) is skipped - MemAvailable alone, as with no cgroup at all.
        const std::filesystem::path mroot = root / "memory";
        auto path = (mroot / v1_path.substr(v1_path.rfind('/', 0) == 0 ? 1 : 0)).lexically_normal();
        while (path.string().rfind(mroot.string(), 0) == 0) {
            std::ifstream limit_file(path / "memory.limit_in_bytes"), usage_file(path / "memory.usage_in_bytes");
            uint64_t cap = 0, usage = 0;
            if (limit_file >> cap && usage_file >> usage && cap < (1ull << 62)) {
                bytes = std::min(bytes, usage < cap ? cap - usage : 0);
                m.cgroup_limit = std::min(m.cgroup_limit, cap);
            }
            if (path == mroot) break;
            path = path.parent_path();
        }
    }
    m.available = bytes;
    return true;
#elif defined(__APPLE__)
    // macOS: free + inactive + purgeable pages (what the OS can hand out without swapping), as vm_stat reports them
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page_bytes = 0;
    if (host_page_size(mach_host_self(), &page_bytes) != KERN_SUCCESS ||
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &count) != KERN_SUCCESS) return false;
    m.available = ((uint64_t) vs.free_count + vs.inactive_count + vs.purgeable_count) * (uint64_t) page_bytes;
    return m.available > 0;
#else
    (void) meminfo; (void) self_cgroup; (void) cgroup_root;
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    const long page_bytes = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_bytes <= 0 ||
        (uint64_t) pages > std::numeric_limits<uint64_t>::max() / (uint64_t) page_bytes) return false;
    m.available = (uint64_t) pages * (uint64_t) page_bytes;
    return m.available > 0;
#endif
}

}  // namespace detail

}  // namespace strata::core
