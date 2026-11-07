#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::serve {

/// Associates model-generated protocol records with the request being processed on this thread.
/// Model implementations share this scope so their statistics keep the same framing as token events.
class RequestOutput {
public:
    explicit RequestOutput(const std::string& id);
    ~RequestOutput();
    RequestOutput(const RequestOutput&) = delete;
    RequestOutput& operator=(const RequestOutput&) = delete;
private:
    std::string previous_;
};

/// Preserves outer generation framing while the command loop answers a nested inspection.
/// Callers finish any fragmented response line before entering this scope.
class InspectionOutput {
public:
    InspectionOutput();
    ~InspectionOutput();
    InspectionOutput(const InspectionOutput&) = delete;
    InspectionOutput& operator=(const InspectionOutput&) = delete;
private:
    std::string request_, command_;
};

/// Configures the shared adapter used by engines that still execute legacy command handlers.
void configure_legacy_requests(int batch_capacity);
/// Opens ASK reply framing and adapts RUN to legacy handlers without exposing placement.
bool translate_protocol_command(std::string& line);
/// Ends request and command output scopes before the next command-loop operation.
void clear_output_request();
/// Gets running legacy placements for targeted cancellation at a safe engine boundary.
std::vector<std::pair<int, std::string>> get_legacy_requests();
/// Collects completed identities so Input can discard their cancellation records.
std::vector<std::string> take_finished_legacy_requests();

/// Gets the request whose synchronous model callbacks are currently running.
const std::string& get_output_request();
/// Frames request events and command replies; preserves fragmented writes until their newline.
int print_response(const char* format, ...);

}  // namespace strata::serve
