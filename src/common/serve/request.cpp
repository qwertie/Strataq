// strata/common/serve/request.cpp - see request.hpp.
#include "strata/common/serve/request.hpp"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace strata::serve {

namespace {
void read_keys(const char*& p, std::vector<std::pair<std::string, std::string>>& keys);
}

bool parse_request_line(const std::string& line, RequestLine& out, bool keys_after_file) {
    out = RequestLine{};
    const bool run = line.rfind("RUN ", 0) == 0 || line.rfind("RUNI ", 0) == 0;
    out.batch = !run && line.rfind("BGEN", 0) == 0;
    const char* p = line.c_str() + (out.batch ? 1 : 0);
    if (run) {
        out.pictures = line.rfind("RUNI ", 0) == 0;
        p += out.pictures ? 5 : 4;
        const char* end = std::strchr(p, ' ');
        if (!end || end == p)
            return false;
        out.request_id.assign(p, (size_t) (end - p));
        if (!is_protocol_identity(out.request_id))
            return false;
        p = end + 1;
    } else {
        if (std::strncmp(p, "GENI ", 5) == 0)
            out.pictures = true;
        else if (std::strncmp(p, "GEN ", 4) != 0)
            return false;
        p += out.pictures ? 5 : 4;
    }
    char* e = nullptr;
    if (out.batch) {
        out.slot = std::strtol(p, &e, 10);
        p = e;
    }
    out.max_new = std::strtoll(p, &e, 10);
    out.has_max_new = e != p;
    p = e;
    read_keys(p, out.keys);
    if (out.pictures) {
        const char* gap = std::strchr(p, ' ');
        if (gap != nullptr) {
            out.file.assign(p, (size_t) (gap - p));
            p = gap;
            if (keys_after_file)
                read_keys(p, out.keys);
        }
    }
    out.ids = p;
    return true;
}

bool is_protocol_identity(const std::string& id) {
    if (id.empty())
        return false;
    for (char c : id)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
              (c >= 'A' && c <= 'Z') || c == '_' || c == '-'))
            return false;
    return true;
}

bool read_sampling_key(const std::string& key, const std::string& value, Sampling& s) {
    const char* v = value.c_str();
    if (key == "temperature") s.temperature = std::strtof(v, nullptr);
    else if (key == "top_p") s.top_p = std::strtof(v, nullptr);
    else if (key == "top_k") s.top_k = std::atoi(v);
    else if (key == "min_p") s.min_p = std::strtof(v, nullptr);
    else if (key == "penalty_last_n") s.penalty_last_n = std::atoi(v);
    else if (key == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
    else if (key == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
    else if (key == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
    else if (key == "seed") s.seed = std::strtoull(v, nullptr, 10);
    else return false;
    return true;
}

std::vector<uint64_t> parse_hex_list(const std::string& text) {
    std::vector<uint64_t> out;
    const char* q = text.c_str();
    while (*q != '\0') {
        char* qe = nullptr;
        const unsigned long long v = std::strtoull(q, &qe, 16);
        if (qe == q || (*qe != ',' && *qe != '\0')) return {};
        out.push_back((uint64_t) v);
        q = *qe == ',' ? qe + 1 : qe;
    }
    return out;
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

namespace {
/// Appends the key=value words at `p` to `keys`, up to the first word without '=', and leaves `p` at that word.
void read_keys(const char*& p, std::vector<std::pair<std::string, std::string>>& keys) {
    for (;;) {
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string word(start, (size_t) (p - start));
        const size_t eq = word.find('=');
        if (word.empty() || eq == std::string::npos) {
            p = start;
            return;
        }
        keys.emplace_back(word.substr(0, eq), word.substr(eq + 1));
    }
}
}  // namespace

}  // namespace strata::serve
