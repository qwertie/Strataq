// How much of a prompt a kept conversation continues from (resume_point.hpp): whole, back at a position it can go
// back to (a re-rendered history goes back to its anchor), or not at all.
#include "strata/common/state/resume_point.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::core::ConversationImageKey;
using strata::core::get_image_match_end;
using strata::core::get_resume_tokens;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}
/// Gets `n` tokens from `first` on, then `then` tokens from 90000 on (where a prompt leaves the conversation).
std::vector<int64_t> get_prompt(int32_t first, size_t n, size_t then = 0) {
    std::vector<int64_t> v;
    for (size_t i = 0; i < n; ++i)
        v.push_back(first + (int64_t) i);
    for (size_t i = 0; i < then; ++i)
        v.push_back(90000 + (int64_t) i);
    return v;
}
}  // namespace

int main() {
    // a conversation of 3,000 tokens; its checkpoints: its last prompt's end and the position before it (2,950 and
    // 2,949) and the anchor (1,000)
    std::vector<int32_t> chat;
    for (int32_t i = 0; i < 3000; ++i)
        chat.push_back(1000 + i);
    const std::vector<int64_t> points = {2950, 2949, 1000};
    check(get_resume_tokens(chat, points, get_prompt(1000, 3000, 40)) == 3000, "the next turn continues it whole");
    check(get_resume_tokens(chat, {}, get_prompt(1000, 3000, 40)) == 3000, "whole, without checkpoints too");
    check(get_resume_tokens(chat, points, get_prompt(1000, 3000)) == 2950,
          "the same prompt again leaves its last token to read: back to the latest checkpoint");
    check(get_resume_tokens(chat, {}, get_prompt(1000, 3000)) == 0, "the same prompt without checkpoints: none");
    check(get_resume_tokens(chat, points, get_prompt(1000, 2960, 30)) == 2950, "a retried turn: its prompt's end");
    check(get_resume_tokens(chat, points, get_prompt(1000, 1200, 30)) == 1000,
          "a history rendered again without its earlier reasoning: back to the anchor");
    check(get_resume_tokens(chat, points, get_prompt(1000, 1000, 30)) == 1000, "parting right at the anchor");
    check(get_resume_tokens(chat, points, get_prompt(1000, 999, 30)) == 0, "parting before every checkpoint: none");
    check(get_resume_tokens(chat, points, get_prompt(1000, 2000)) == 1000, "a shorter prompt: a checkpoint within it");
    check(get_resume_tokens(chat, {0, -1, 4000}, get_prompt(1000, 2000, 10)) == 0,
          "no checkpoint at 0, below it or past the conversation");
    check(get_resume_tokens(chat, points, get_prompt(5000, 100)) == 0, "another conversation: none");
    check(get_resume_tokens({}, points, get_prompt(1000, 100)) == 0, "an empty conversation: none");
    check(get_resume_tokens(chat, points, std::vector<int64_t>{}) == 0, "an empty prompt: none");
    check(get_resume_tokens(chat, points, std::vector<int>{1000}) == 0, "a prompt of one token: none");

    // pictures: their tokens are equal placeholders, so where the pictures differ bounds the match
    const std::vector<ConversationImageKey> one = {{500, 7}}, two = {{500, 7}, {2990, 9}};
    check(get_image_match_end(one, one) == SIZE_MAX && get_image_match_end({}, {}) == SIZE_MAX, "the same pictures");
    check(get_image_match_end(one, two) == 2990 && get_image_match_end(two, one) == 2990,
          "a picture added after the others: up to its start");
    check(get_image_match_end(one, {{500, 8}}) == 500, "another picture in the same place: up to it");
    check(get_image_match_end(one, {{400, 7}}) == 400, "a picture moved: up to the earlier start");
    check(get_image_match_end({}, one) == 500, "text, then a prompt with a picture: up to the picture");
    check(get_resume_tokens(chat, points, get_prompt(1000, 3000, 40), get_image_match_end(one, two)) == 2950,
          "the next turn adds a picture inside the conversation: back to the latest checkpoint before it");
    check(get_resume_tokens(chat, points, get_prompt(1000, 3000, 40), 3000) == 3000,
          "the next turn adds a picture after the conversation: whole");
    check(get_resume_tokens(chat, points, get_prompt(1000, 3000, 40), 999) == 0,
          "pictures that differ before every checkpoint: none");
    std::printf("resume_point_test: %d checks passed\n", checks);
    return 0;
}
