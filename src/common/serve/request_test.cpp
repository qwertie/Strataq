// The request lines of --serve (request.hpp), split as strata_common/engine_client.py writes them.
#include "strata/common/serve/request.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace strata::serve;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
}  // namespace

int main() {
    RequestLine r;
    check(parse_request_line("RUN chat_a 64 temperature=0.7 1,2,3", r), "RUN is a request");
    check(r.request_id == "chat_a" && !r.batch && !r.pictures && r.max_new == 64, "RUN has no placement");
    check(parse_request_line("RUNI r2 3 /tmp/p.bin 7,8", r) && r.pictures && r.request_id == "r2",
          "RUNI keeps the same request identity");
    check(!parse_request_line("RUN bad/id 4 1,2", r), "request identities are single protocol-safe words");
    check(parse_request_line("GEN 64 temperature=0.7 seed=5 1,2,3", r), "GEN is a request");
    check(!r.batch && !r.pictures && r.max_new == 64, "GEN's header");
    check(r.keys.size() == 2 && r.keys[0].first == "temperature" && r.keys[1].second == "5", "GEN's keys in order");
    check(r.ids == "1,2,3" && r.file.empty(), "GEN's ids");

    check(parse_request_line("BGENI 3 1 img_keys=a,ff /tmp/p.bin 7,8", r), "BGENI is a request");
    check(r.batch && r.pictures && r.slot == 3 && r.max_new == 1, "BGENI's header");
    check(r.file == "/tmp/p.bin" && r.ids == " 7,8", "BGENI's file and ids");
    check(parse_hex_list(r.keys[0].second) == std::vector<uint64_t>{0xa, 0xff}, "img_keys");
    check(parse_hex_list("a,x").empty(), "a malformed img_keys is empty");

    check(parse_request_line("GENI 5 /tmp/p.bin", r) && r.file.empty() && r.ids == "/tmp/p.bin",
          "GENI without ids has no file");
    check(parse_request_line("GEN x", r) && r.max_new == 0 && !r.has_max_new, "no max_new reads as 0");
    check(parse_request_line("GENI 5 a=1 /tmp/p.bin b=2 7,8", r, true) && r.has_max_new && r.keys.size() == 2 &&
          r.keys[1].first == "b" && r.file == "/tmp/p.bin" && r.ids == "7,8", "keys after the file");
    check(parse_request_line("GENI 5 /tmp/p.bin b=2 7,8", r) && r.keys.empty() && r.ids == " b=2 7,8",
          "without keys_after_file the file ends the keys");
    check(!parse_request_line("GENX 5 1", r) && !parse_request_line("BGEN", r) && !parse_request_line("QUIT", r),
          "other lines are not requests");

    Sampling s;
    check(read_sampling_key("top_k", "40", s) && s.top_k == 40, "a sampling key");
    check(!read_sampling_key("pin", "4", s), "another key");

    LogprobRequest lp;
    check(read_logprob_key("logprobs", "2", lp) && lp.top == 2 && !read_logprob_key("top_k", "4", lp), "logprobs");
    const float logits[4] = {0.0f, 2.0f, 1.0f, 0.0f};   // log-softmax: -2.4938 -0.4938 -1.4938 -2.4938
    check(format_logprob_line(logits, 4, 2, lp) == "LP -1.493812 1:-0.493812 2:-1.493812\n", "the top two");
    check(format_logprob_line(nullptr, 4, 2, lp) == "LP nan\n" && format_logprob_line(logits, 4, 4, lp) == "LP nan\n",
          "lost logits and a token past the vocabulary");
    check(read_logprob_key("allowed", "3,9,,2", lp) && lp.allowed == std::vector<int32_t>{3, 9, 2}, "allowed");
    check(format_logprob_line(logits, 4, 2, lp) == "LP -1.493812 3:-2.493812\n",
          "allowed: the picked token and an id past the vocabulary are left out");

    RestorePointRequest rp;
    check(rp.get_shared() == -1 && !read_restore_point_key("top_k", "4", rp), "no restore points");
    check(read_restore_point_key("marks", "3000,,0,4000", rp) && rp.marks == std::vector<int64_t>{3000, 4000} &&
          read_restore_point_key("tail", "5000", rp) && rp.tail == 5000, "marks and tail");
    check(read_restore_point_key("pin", "255", rp) && rp.get_shared() == -1 &&
          read_restore_point_key("pin", "256", rp) && rp.get_shared() == 256, "a pin of 256 tokens or more");
    check(read_restore_point_key("shared", "200", rp) && rp.get_shared() == 200, "shared before pin");

    RowLogprobs rows;
    rows.start({}, 5);
    rows.note(5, 1, 4, [&](float*) -> bool { return check(false, "not asked: no copy"), false; });
    check(rows.take(5).empty(), "not asked: no line");
    rows.start({1, {}}, 5);
    const auto copy = [&](float* out) { return std::copy(logits, logits + 4, out), true; };
    rows.note(4, 1, 4, copy);   // a prompt row
    rows.note(6, 2, 4, copy);   // a rejected guess, then the row that replaced it
    rows.note(6, 1, 4, copy);
    rows.note(5, 1, 4, [](float*) { return false; });
    check(rows.take(4) == "LP nan\n", "a prompt row gets no line");
    check(rows.take(5) == "LP nan\n", "lost logits");
    check(rows.take(6) == "LP -0.493812 1:-0.493812\n" && rows.take(6) == "LP nan\n",
          "the last pick at a position wins, and is taken once");
    std::printf("request_test: %d checks passed\n", checks);
    return 0;
}
