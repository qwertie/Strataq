// src/common/state/pause_file.cpp - the file of a paused conversation, which is also a saved session (pause_file.hpp).
#include "strata/common/state/pause_file.hpp"
#include "strata/common/state/state_bytes.hpp"

#include "strata/base/gpu/on_device.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {

/// The file's first bytes; then the header's size, the header and its hash (uint64_t each).
constexpr char kMagic[8] = {'S', 'T', 'R', 'P', 'A', 'U', 'S', '2'};
constexpr uint64_t kHashSeed = 0x5041555345484452ull;   // "PAUSEHDR"

bool flush_to_drive(std::FILE* f);

std::FILE* open_file(const std::string& path, const char* mode);

bool seek_file(std::FILE* file, uint64_t at);

bool hash_file(std::FILE* file, uint64_t bytes, uint64_t& hash);

bool publish_file(PauseStream& file, const std::string& temporary, const std::string& path, bool ok,
                  std::string& err);

}  // namespace

bool write_pause_file(const std::string& path, const SessionFileIdentity& identity, const PauseHeader& head,
                      const std::vector<PauseSection>& sections, std::string& err) {
    std::vector<uint8_t> body;
    body.reserve(sizeof identity + 4 * sizeof(uint64_t) + head.live.size() * sizeof(head.live[0]) +
                 head.points.size() * sizeof(head.points[0]) + head.state.size() +
                 head.cut.size() * sizeof(head.cut[0]) +
                 (!head.images.empty() || head.last_tokens_at != 0
                      ? sizeof(uint64_t) + head.images.size() * sizeof(head.images[0]) : 0) +
                 (head.last_tokens_at != 0 ? sizeof head.last_tokens_at : 0));
    append_state_bytes(body, &identity, sizeof identity);
    append_state_vector(body, head.live);
    append_state_vector(body, head.points);
    append_state_vector(body, head.state);
    append_state_vector(body, head.cut);
    if (!head.images.empty() || head.last_tokens_at != 0)
        append_state_vector(body, head.images);
    if (head.last_tokens_at != 0)
        append_state_bytes(body, &head.last_tokens_at, sizeof head.last_tokens_at);
    const uint64_t size = body.size(), hash = session_hash64(body.data(), body.size(), kHashSeed);
    const std::string temporary = path + ".tmp";
    PauseStream f;
    f.reset(open_file(temporary, "w+bx"));
    if (!f) {
        err = "cannot create " + temporary;
        return false;
    }
    std::string why;
    bool ok = f != nullptr && std::fwrite(kMagic, 1, sizeof kMagic, f.get()) == sizeof kMagic &&
              std::fwrite(&size, sizeof size, 1, f.get()) == 1 &&
              std::fwrite(body.data(), 1, body.size(), f.get()) == body.size() &&
              std::fwrite(&hash, sizeof hash, 1, f.get()) == 1;
    for (const PauseSection& s : sections) {
        const OnDevice on(s.device);
        ok = ok && s.pool != nullptr && write_conversation_section(f.get(), *s.pool, s.id, why, s.span);
    }
    ok = ok && std::fflush(f.get()) == 0;
    std::error_code ec;
    const uint64_t bytes = std::filesystem::file_size(std::filesystem::u8path(temporary), ec);
    uint64_t checksum = 0;
    ok = ok && !ec && hash_file(f.get(), bytes, checksum) && seek_file(f.get(), bytes) &&
         std::fwrite(&checksum, sizeof checksum, 1, f.get()) == 1;
    err = why;
    return publish_file(f, temporary, path, ok, err);
}

bool copy_pause_file(const std::string& from, const std::string& to, std::string& err) {
    err.clear();
    PauseStream src, dst;
    src.reset(open_file(from, "rb"));
    const std::string temporary = to + ".tmp";
    dst.reset(open_file(temporary, "wbx"));
    if (!dst) {
        err = "cannot create " + temporary;
        return false;
    }
    std::vector<uint8_t> buffer(1u << 20);
    bool ok = src != nullptr;
    while (ok) {
        const size_t n = std::fread(buffer.data(), 1, buffer.size(), src.get());
        ok = std::fwrite(buffer.data(), 1, n, dst.get()) == n && !std::ferror(src.get());
        if (n < buffer.size())
            break;
    }
    return publish_file(dst, temporary, to, ok, err);
}

SessionError open_pause_file(const std::string& path, const SessionFileIdentity& identity, uint64_t max_tokens,
                             PauseStream& file, PauseHeader& head, std::string& err) {
    const auto refuse = [&](SessionError kind, const std::string& why) {
        file.reset();
        err = path + ": " + why;
        return kind;
    };
    std::error_code ec;
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(std::filesystem::u8path(path), ec)))
        return refuse(SessionError::invalid, "a saved session cannot be a link");
    const uint64_t file_bytes = std::filesystem::file_size(std::filesystem::u8path(path), ec);
    file.reset(open_file(path, "rb"));
    if (ec || file == nullptr)
        return refuse(SessionError::io, "cannot be opened");
    uint64_t checksum = 0, stored_checksum = 0;
    if (file_bytes < 32 || !hash_file(file.get(), file_bytes - sizeof checksum, checksum) ||
        std::fread(&stored_checksum, sizeof stored_checksum, 1, file.get()) != 1 || checksum != stored_checksum ||
        !seek_file(file.get(), 0))
        return refuse(SessionError::invalid, "damaged or truncated");
    char magic[sizeof kMagic] = {};
    uint64_t size = 0, hash = 0;
    if (std::fread(magic, 1, sizeof magic, file.get()) != sizeof magic || std::memcmp(magic, kMagic, sizeof magic) != 0 ||
        std::fread(&size, sizeof size, 1, file.get()) != 1 || size > file_bytes - 32)
        return refuse(SessionError::invalid, "not a saved session");
    std::vector<uint8_t> body((size_t) size);
    if (std::fread(body.data(), 1, body.size(), file.get()) != body.size() ||
        std::fread(&hash, sizeof hash, 1, file.get()) != 1 || hash != session_hash64(body.data(), body.size(), kHashSeed))
        return refuse(SessionError::invalid, "damaged");
    StateReader r{body};
    SessionFileIdentity bound;
    PauseHeader read;
    if (!r.take(&bound, sizeof bound) || !r.take_vector(read.live) || !r.take_vector(read.points) ||
        !r.take_vector(read.state) ||
        !r.take_vector(read.cut) ||
        (r.at != body.size() && !r.take_vector(read.images)) ||
        (r.at != body.size() && !r.take(&read.last_tokens_at, sizeof read.last_tokens_at)) || r.at != body.size())
        return refuse(SessionError::invalid, "damaged");
    if (bound.model != identity.model || bound.config != identity.config)
        return refuse(SessionError::invalid, "saved by another model or with other settings");
    if (read.live.empty() || read.live.size() > max_tokens)
        return refuse(SessionError::invalid, std::to_string(read.live.size()) + " tokens: more than this engine holds");
    for (const int64_t point : read.points)
        if (point < 1 || point > (int64_t) read.live.size())
            return refuse(SessionError::invalid, "damaged");
    head = std::move(read);
    return SessionError::none;
}

namespace {

/// Flushes `f` to the drive.
bool flush_to_drive(std::FILE* f) {
#if defined(_WIN32)
    return std::fflush(f) == 0 && _commit(_fileno(f)) == 0;
#else
    return std::fflush(f) == 0 && fsync(fileno(f)) == 0;
#endif
}

/// Opens UTF-8 paths from the engine protocol.
std::FILE* open_file(const std::string& path, const char* mode) {
#if defined(_WIN32)
    const std::wstring wide_mode(mode, mode + std::strlen(mode));
    return _wfopen(std::filesystem::u8path(path).c_str(), wide_mode.c_str());
#else
    return std::fopen(path.c_str(), mode);
#endif
}

/// Seeks without the 2 GiB limit of Windows' fseek.
bool seek_file(std::FILE* file, uint64_t at) {
#if defined(_WIN32)
    return _fseeki64(file, at, SEEK_SET) == 0;
#else
    return fseeko(file, (off_t) at, SEEK_SET) == 0;
#endif
}

/// Hashes the first `bytes` bytes with bounded host memory, leaving the stream after them.
bool hash_file(std::FILE* file, uint64_t bytes, uint64_t& hash) {
    if (!seek_file(file, 0))
        return false;
    SessionHasher h(kHashSeed);
    std::vector<uint8_t> buffer(1u << 20);
    while (bytes > 0) {
        const size_t n = (size_t) std::min<uint64_t>(bytes, buffer.size());
        if (std::fread(buffer.data(), 1, n, file) != n)
            return false;
        h.update(buffer.data(), n);
        bytes -= n;
    }
    hash = h.digest();
    return true;
}

/// Publishes a completed temporary file, preserving the previous destination on failure.
bool publish_file(PauseStream& file, const std::string& temporary, const std::string& path, bool ok,
                  std::string& err) {
    ok = ok && flush_to_drive(file.get());
    if (file)
        ok = std::fclose(file.release()) == 0 && ok;
    std::error_code ec;
    if (ok) {
#if defined(_WIN32)
        ok = MoveFileExW(std::filesystem::u8path(temporary).c_str(), std::filesystem::u8path(path).c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        std::filesystem::rename(temporary, path, ec);
        ok = !ec;
#endif
    }
    if (!ok) {
        std::filesystem::remove(std::filesystem::u8path(temporary), ec);
        err = "could not write " + path + (err.empty() ? "" : ": " + err);
    }
    return ok;
}

}  // namespace
}  // namespace strata::core
