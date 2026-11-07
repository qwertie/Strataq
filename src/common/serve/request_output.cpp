#include "strata/common/serve/request_output.hpp"
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <vector>
#include <map>
#include <sstream>

namespace strata::serve {
namespace {
// Model callbacks do not accept a transport parameter; this scope is confined to their invoking thread.
thread_local std::string request_id, command_id;
std::mutex output_mutex;
/// Retains admission statistics until the legacy handler finishes the request.
struct LegacyRequest {
    std::string id, stats;
};
/// Hides handler placement and admission handshakes from the request-ID protocol.
struct LegacyRequests {
    int capacity = -1;
    int admission = -1;
    std::map<int, LegacyRequest> running;
    std::vector<std::string> finished;
};
thread_local LegacyRequests legacy;
thread_local std::string pending;

void emit_line(const std::string& line);
void finish_legacy(int place);
std::string merge_legacy_done(const std::string& stats, const std::string& completion);
}

RequestOutput::RequestOutput(const std::string& id) : previous_(request_id) { request_id = id; }
RequestOutput::~RequestOutput() { request_id = previous_; }
const std::string& get_output_request() { return request_id; }

void configure_legacy_requests(int batch_capacity) { legacy.capacity = batch_capacity; }
void clear_output_request() { request_id.clear(); command_id.clear(); }

bool translate_protocol_command(std::string& line) {
    if (line.rfind("ASK ", 0) == 0) {
        const size_t end = line.find(' ', 4);
        const std::string id = line.substr(4, end == std::string::npos ? end : end - 4);
        if (end == std::string::npos || !is_protocol_identity(id)) {
            print_response("ERR malformed ASK command\n");
            return false;
        }
        command_id = id;
        line.erase(0, end + 1);
        if (line.rfind("RUN ", 0) == 0 || line.rfind("RUNI ", 0) == 0) {
            print_response("ERR generation uses RUN directly\n");
            return false;
        }
    }
    if (legacy.capacity < 0 || (line.rfind("RUN ", 0) != 0 && line.rfind("RUNI ", 0) != 0))
        return true;
    RequestLine request;
    if (!parse_request_line(line, request) || request.request_id.empty()) {
        print_response("ERR malformed RUN request\n");
        return false;
    }
    request_id = request.request_id;
    for (const auto& [place, active] : legacy.running) {
        if (active.id == request_id) {
            legacy.finished.push_back(request_id);
            print_response("ERR duplicate request identity\n");
            return false;
        }
    }
    const int capacity = std::max(1, legacy.capacity);
    int place = 0;
    while (place < capacity && legacy.running.contains(place))
        ++place;
    if (place == capacity) {
        legacy.finished.push_back(request_id);
        print_response("ERR request capacity is occupied\n");
        return false;
    }
    legacy.running.emplace(place, LegacyRequest{request_id, {}});
    legacy.admission = place;
    print_response("STARTED\n");
    const size_t tail = line.find(' ', line.find(' ') + 1);
    line = (legacy.capacity > 0 ? std::string(request.pictures ? "BGENI " : "BGEN ") +
            std::to_string(place) : std::string(request.pictures ? "GENI" : "GEN")) + line.substr(tail);
    return true;
}

std::vector<std::pair<int, std::string>> get_legacy_requests() {
    std::vector<std::pair<int, std::string>> requests;
    if (legacy.capacity > 0)
        for (const auto& [place, active] : legacy.running)
            requests.emplace_back(place, active.id);
    return requests;
}

std::vector<std::string> take_finished_legacy_requests() {
    std::vector<std::string> finished;
    finished.swap(legacy.finished);
    return finished;
}

int print_response(const char* format, ...) {
    va_list args, copy;
    va_start(args, format);
    va_copy(copy, args);
    const int n = std::vsnprintf(nullptr, 0, format, copy);
    va_end(copy);
    if (n < 0) {
        va_end(args);
        return n;
    }
    std::vector<char> bytes((size_t) n + 1);
    std::vsnprintf(bytes.data(), bytes.size(), format, args);
    va_end(args);
    pending.append(bytes.data(), (size_t) n);
    for (size_t end; (end = pending.find('\n')) != std::string::npos;) {
        const std::string line = pending.substr(0, end);
        pending.erase(0, end + 1);
        emit_line(line);
    }
    return n;
}

namespace {
void finish_legacy(int place) {
    const auto found = legacy.running.find(place);
    if (found != legacy.running.end()) {
        legacy.finished.push_back(found->second.id);
        legacy.running.erase(found);
    }
}

std::string merge_legacy_done(const std::string& stats, const std::string& completion) {
    std::istringstream source(stats), completed(completion);
    std::vector<std::string> fields;
    for (std::string field; source >> field;)
        fields.push_back(std::move(field));
    if (fields.size() < 6 || !(completed >> fields[1] >> fields[5] >> fields[4]))
        return "ERR missing generation statistics";
    std::string result;
    for (const std::string& field : fields) {
        if (!result.empty())
            result += ' ';
        result += field;
    }
    return result;
}

void emit_line(const std::string& line) {
    std::string id = request_id, payload = line;
    const bool telemetry = line.rfind("BSTAT", 0) == 0 || line.rfind("PSTAT", 0) == 0;
    if (legacy.capacity >= 0) {
        std::istringstream input(line);
        std::string kind;
        input >> kind;
        if (kind == "BT" || kind == "BDONE" || kind == "BADM") {
            int place = -1;
            input >> place;
            const auto found = legacy.running.find(place);
            if (found != legacy.running.end()) {
                id = found->second.id;
                std::string rest;
                std::getline(input >> std::ws, rest);
                if (kind == "BT")
                    payload = "T " + rest;
                else if (kind == "BDONE") {
                    payload = merge_legacy_done(found->second.stats, rest);
                    finish_legacy(place);
                } else {
                    const std::string stats = found->second.stats;
                    request_id.clear();
                    legacy.admission = -1;
                    if (rest != "0")
                        return;
                    payload = stats.empty() ? "ERR missing generation statistics" : stats;
                    finish_legacy(place);
                }
            }
        } else if (!id.empty() && legacy.admission >= 0) {
            if (kind == "DONE" && legacy.capacity > 0) {
                legacy.running.at(legacy.admission).stats = line;
                return;
            } else if (kind == "DONE" || kind == "ERR") {
                finish_legacy(legacy.admission);
                legacy.admission = -1;
                request_id.clear();
            }
        }
        if (telemetry)
            id.clear();
    }
    std::lock_guard<std::mutex> lock(output_mutex);
    if (!id.empty())
        std::printf("R %s ", id.c_str());
    else if (!telemetry && !command_id.empty())
        std::printf("C %s ", command_id.c_str());
    std::fwrite(payload.data(), 1, payload.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}
}

}  // namespace strata::serve
