// include/strata/common/experts/ram_tier.hpp - the expert store's RAM tier: how much RAM it may take (MemAvailable,
// cgroup limits, NUMA nodes, Windows commit) and which experts its compact RAM copy (the cache complement) holds.
// CPU-only, so that the arithmetic can be tested without a GPU; FileExpertSource (src/core/expert_source.cpp) uses it.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

namespace detail {

/// Sentinel used by the pure complement planner for a blob that remains in the mmap fallback.
inline constexpr uint64_t kNoCacheComplement = ~uint64_t{0};

/// Required cgroup-v2 usage counters for the conservative cache-reclaim allowance.
struct CgroupMemoryStat {
    uint64_t current = 0;
    uint64_t inactive_file = 0;
    uint64_t file_dirty = 0;
    uint64_t file_writeback = 0;
    bool valid = false;
    uint64_t file = 0;   ///< 149: memory.stat "file" (active + inactive page cache); 0 when not reported
};

/// Calculate additional bytes under a finite cgroup limit after reclaiming the group's clean page cache, as
/// MemAvailable counts it on a bare host: every clean file page (active and inactive: memory.stat "file", else
/// inactive_file), less a low-watermark share the kernel keeps (min(half the clean cache, limit / 64)).  A process
/// whose own reads (a GPU cache fill, a read-ahead) left gigabytes of active cache in its group was judged short of
/// RAM before (149: 1.2 GiB "available" of a 32 GiB group, the file tier then switched to unbuffered reads).
/// Returns false when the required memory.stat counters were unavailable.
bool cgroup_available_bytes(uint64_t limit, const CgroupMemoryStat& stat, uint64_t& bytes);

/// #633: the RAM this process can get.  `available`: MemAvailable (Windows: the available physical memory), lowered
/// to the room under the tightest cgroup limit; `cgroup_limit`: that tightest limit itself (v2 memory.max of the group
/// and its ancestors, or v1 memory.limit_in_bytes), ~0 when there is none - what a container can never exceed.
struct HostMemory {
    uint64_t available = 0;
    uint64_t cgroup_limit = ~uint64_t{0};
    uint64_t commit = ~uint64_t{0};   ///< available Windows commit capacity; ~0 when not reported
};

/// Linux reads `meminfo`, `self_cgroup` and the cgroup tree under `cgroup_root` (the parameters are for tests; the
/// defaults are the real files).  cgroup v2 as before (an unreadable limit of a group that has one fails); cgroup v1's
/// memory controller (`<root>/memory/<path>`: memory.limit_in_bytes - memory.usage_in_bytes); no cgroup line at all is
/// MemAvailable alone.  False when the RAM cannot be determined.
bool host_available_memory(HostMemory& m, const std::string& meminfo = "/proc/meminfo",
                           const std::string& self_cgroup = "/proc/self/cgroup",
                           const std::string& cgroup_root = "/sys/fs/cgroup");

/// #4051: wait for the RAM an arena needs to come back.  An engine that was still exiting (or a program that was ending)
/// holds memory a few seconds longer than its process; loading into that shortfall sent a 60 GiB host into reclaim for
/// good.  Calls `sample(available)` (false: the RAM cannot be read; the wait ends) every `step_s` seconds, through
/// `sleep(step_s)`, until `available >= need` or `max_wait_s` seconds have gone.  `waited` gets the seconds waited,
/// `available` the last reading.  True when the RAM is there.  `progress(waited, available)` is called about every
/// `note_s` seconds while waiting.  `max_wait_s <= 0` is no wait at all.
bool wait_for_ram(uint64_t need, double max_wait_s, double step_s, double note_s,
                  const std::function<bool(uint64_t&)>& sample, const std::function<void(double)>& sleep,
                  const std::function<void(double, uint64_t)>& progress, double& waited, uint64_t& available);

/// Bound a resident budget by RAM and commit capacity after headroom; leave 256 MiB more when clamping.
/// Pass UINT64_MAX for commit when the platform does not report it.
uint64_t clamp_resident_budget(uint64_t requested, uint64_t physical, uint64_t commit, uint64_t headroom);

/// #1250: one cudaHostAlloc of the whole page-locked complement has no way back when the kernel does not give the
/// pages as fast as the driver takes them (the process is OOM-killed, even with MemAvailable high: clean file cache
/// that cannot be reclaimed at that moment).  `free_now` is MemFree (pages that are really free), `bytes` the
/// complement, `reserve` what must stay.  True when the pages would have to come from reclaimed cache.
bool pin_depends_on_reclaim(uint64_t free_now, uint64_t bytes, uint64_t reserve);
/// The paced registration takes the next `step` bytes only while the RAM available after it (cgroup limits included)
/// is still above `reserve`.
bool pin_step_fits(uint64_t available, uint64_t step, uint64_t reserve);
/// Parses `key` ("MemFree") of a /proc/meminfo text; 0 when absent.
uint64_t meminfo_bytes(const std::string& meminfo_text, const std::string& key);

/// 0.1.42 (NUMA report): the MemFree to judge the page-locked complement by.  `global_free` is the whole host's MemFree;
/// `node_free` each NUMA node's own (/sys/devices/system/node/node*/meminfo, empty or one entry on a single-node host).
/// A dual-socket host whose second node is full of another model's file cache has plenty free in total but none where
/// the driver may allocate: with two or more nodes the figure is the smaller of the total and (node count x the
/// SMALLEST node's free), i.e. what an even spread would find.  Only lowers the figure, so the pages are then taken in
/// steps (which reclaim cache as they fault) instead of in one driver call; it never refuses and never caps.
uint64_t numa_pin_free(uint64_t global_free, const std::vector<uint64_t>& node_free);
/// Byte ranges [first, second) of slice `slice` of 2 inside one complement slot at arena offset `slot_off`, whose roles
/// (gate, up, down) are `role_bytes[0..2]` long and back to back. Role r's slice-0 bytes are [start_r, start_r + role_r/2),
/// its slice-1 bytes the rest. Page rule (absolute arena offsets): any page holding a slice-0 byte belongs to slice 0,
/// so slice-0 ranges round OUTWARDS (start down, end up) and slice-1 ranges round INWARDS (start up, end down) - this
/// also settles pages shared across role and slot boundaries. Returns three ranges in arena offsets, possibly empty
/// (first == second). Adjacent roles smaller than a page may give overlapping slice-0 ranges: harmless (same node, the
/// caller coalesces them).
std::array<std::pair<uint64_t, uint64_t>, 3> numa_slice_ranges(uint64_t slot_off, const uint64_t role_bytes[3],
                                                               int slice, uint64_t page = 4096);
/// The sentence appended to a "not enough RAM" start failure: how much of the RAM is file cache (Cached + Buffers of
/// `meminfo_text`, mapped pages included) and what to do about it; "" when under 1 GiB of cache (nothing to say).
std::string page_cache_hint(const std::string& meminfo_text);

/// Build compact offsets for experts absent from both the primary GPU cache and an optional second GPU tier.
/// Kept CPU-only so selection and byte accounting can be tested without initializing a GPU.
bool make_cache_complement_plan(
    int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
    const std::vector<std::pair<int32_t, int32_t>>& primary_gpu_pairs,
    const std::vector<std::pair<int32_t, int32_t>>& additional_gpu_pairs,
    std::vector<uint64_t>& offsets, uint64_t& bytes, std::string& err);

/// Resolve one blob through the compact copy when present, otherwise preserve its exact mapped-file fallback.
const uint8_t* cache_complement_blob_or_fallback(
    size_t index, const std::vector<uint64_t>& offsets, const uint8_t* complement_host,
    const uint8_t* mapped_fallback);

/// The resident RAM mode: which GPU-cache slots' experts are kept in RAM too.  The prompt path lends the cache's
/// LAST slots (from `lend_from` on; a short prompt lends only the last few), and a lent slot's expert is streamed
/// from RAM during the prompt and copied back into its slot after it.  `base_bytes` (every expert no slot holds)
/// must fit `budget`; slots are then added from the end down to `lend_from` while they still fit.  Returns the
/// first slot kept in RAM (`slot_bytes.size()` = none), or -1 when `base_bytes` alone exceeds `budget`.
int64_t choose_resident_keep_from(const std::vector<uint64_t>& slot_bytes, uint64_t base_bytes, uint64_t budget,
                                  int64_t lend_from);

/// The adaptive tier swapped `in` into a GPU slot and `out` out of it: `out` takes `in`'s place in the compact copy
/// (the caller copies out's bytes there).  False, and nothing changed, unless `in` is in the copy and `out` is not.
bool exchange_cache_complement(std::vector<uint64_t>& offsets, size_t in, size_t out);

/// A layer split: add each stage's lend region to a plan built from the complement plus every stage's
/// cache (the `additional_gpu_pairs` shape).  Each region lists one stage's (layer, expert) pairs, the
/// pairs in its cache's highest slots first - the slots the prompt path borrows, from the tail in.  A pair
/// already in the plan is passed over; a stage's walk stops at the first pair that does not fit `cap_bytes`
/// and the NEXT stage's walk still runs (the pairs past the cap keep the mapped-file fallback they had).
/// Returns the pairs added, or -1 - leaving the plan untouched - when a pair is outside the geometry or a
/// region repeats one.
int64_t append_stage_lend_regions(int64_t n_layers, int64_t n_expert, const std::vector<uint64_t>& layer_blob_bytes,
                                  std::vector<uint64_t>& offsets, uint64_t& bytes, uint64_t cap_bytes,
                                  const std::vector<std::vector<std::pair<int32_t, int32_t>>>& regions,
                                  std::string& err);

}  // namespace detail

}  // namespace strata::core
