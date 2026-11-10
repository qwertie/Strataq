// strata/common/serve/request.cpp - see request.hpp.
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

bool read_logprob_key(const std::string& key, const std::string& value, LogprobRequest& r) {
    if (key == "logprobs") {
        r.top = std::atoi(value.c_str());
        return true;
    }
    if (key != "allowed")
        return false;
    std::istringstream list(value);
    for (std::string one; std::getline(list, one, ',');)
        if (!one.empty())
            r.allowed.push_back((int32_t) std::atoi(one.c_str()));
    return true;
}

std::string format_logprob_line(const float* logits, int64_t vocab, int32_t token, const LogprobRequest& r) {
    if (logits == nullptr || token < 0 || token >= vocab)
        return "LP nan\n";
    const float most = *std::max_element(logits, logits + vocab);
    double sum = 0.0;
    for (int64_t v = 0; v < vocab; ++v)
        sum += std::exp((double) logits[v] - (double) most);
    const double lse = (double) most + std::log(sum);
    char buf[64];
    std::snprintf(buf, sizeof buf, "LP %.6f", (double) logits[token] - lse);
    std::string line = buf;
    const auto add = [&](int32_t id) {
        std::snprintf(buf, sizeof buf, " %d:%.6f", (int) id, (double) logits[id] - lse);
        line += buf;
    };
    if (!r.allowed.empty()) {
        for (const int32_t id : r.allowed)
            if (id != token && id >= 0 && id < vocab)
                add(id);
    } else if (r.top > 0) {
        std::vector<int32_t> order((size_t) vocab);
        std::iota(order.begin(), order.end(), 0);
        const int64_t k = std::min<int64_t>(r.top, vocab);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
        for (int64_t j = 0; j < k; ++j)
            add(order[(size_t) j]);
    }
    return line + "\n";
}

void RowLogprobs::start(LogprobRequest r, int64_t first) {
    request_ = std::move(r);
    first_ = first;
    lines_.clear();
}

void RowLogprobs::note(int64_t pos, int32_t token, int64_t vocab, const std::function<bool(float*)>& copy) {
    if (request_.top < 0 || pos < first_ || token < 0)
        return;
    logits_.resize((size_t) vocab);
    lines_[pos] = format_logprob_line(copy(logits_.data()) ? logits_.data() : nullptr, vocab, token, request_);
}

std::string RowLogprobs::take(int64_t pos) {
    if (request_.top < 0)
        return "";
    const auto found = lines_.find(pos);
    if (found == lines_.end())
        return "LP nan\n";
    std::string line = std::move(found->second);
    lines_.erase(found);
    return line;
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
