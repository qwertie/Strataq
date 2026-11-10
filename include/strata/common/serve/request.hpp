// strata/common/serve/request.hpp - a request line of the server's protocol, split into its parts:
//
//     RUN    <request-id> <max_new> [key=value ...] <id,id,...>
//     RUNI   <request-id> <max_new> [key=value ...] <file> <id,id,...>
//
// Legacy handler forms, internal to the shared adapter or clients of older releases:
//     GEN    <max_new> [key=value ...] <id,id,...>
//     GENI   <max_new> [key=value ...] <file> <id,id,...>         (the prompt's pictures are in <file>)
//     BGEN   <slot> <max_new> [key=value ...] <id,id,...>         (--batch: continue the request in a batch slot)
//     BGENI  <slot> <max_new> [key=value ...] <file> <id,id,...>
//
// The keys every engine reads are the sampling ones (Sampling); the others belong to one engine, and an engine skips
// a key it does not know.  Absent keys keep greedy decoding without penalties.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strata::serve {

/// One request line's parts, as the line gives them: whether they make sense is the engine's to check.
struct RequestLine {
    std::string request_id;    ///< RUN / RUNI identity, independent of execution placement.
    bool batch = false;         ///< BGEN / BGENI
    bool pictures = false;      ///< GENI / BGENI
    long slot = 0;              ///< BGEN's batch slot
    long long max_new = 0;      ///< the most tokens to generate (0 where the line has no number)
    bool has_max_new = false;   ///< whether a number stood where max_new goes
    std::vector<std::pair<std::string, std::string>> keys;   ///< the key=value options, in the line's order
    std::string file;           ///< GENI's picture file; empty when no space and ids follow it
    std::string ids;            ///< the rest of the line: the prompt's token ids, comma-separated
};

/// Checks the single-word identities used by generation and command envelopes.
bool is_protocol_identity(const std::string& id);

/// Splits `line` into `out`; false when it is not a request line.  The options end at the first word without '=';
/// with `keys_after_file` (DeepSeek's lines), more of them may follow GENI's picture file.
bool parse_request_line(const std::string& line, RequestLine& out, bool keys_after_file = false);

/// The sampling options of a request: temperature=F top_p=F top_k=N min_p=F seed=N penalty_last_n=N
/// penalty_repeat=F penalty_freq=F penalty_present=F.
struct Sampling {
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 20;                ///< Qwen's sampler default (its sampled path needs 1..64)
    unsigned long long seed = 0;   ///< 0: the engine picks one
    int penalty_last_n = 0;        ///< how many recent tokens the penalties look at
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
};

/// Stores `value` in `s` if `key` names a sampling option; false for any other key.
bool read_sampling_key(const std::string& key, const std::string& value, Sampling& s);

/// The log probabilities a request asks for, which the engine prints as an LP line after each token's T line:
/// logprobs=K and allowed=a,b,c.
struct LogprobRequest {
    int top = -1;                    ///< K (0..20): how many of the most likely tokens to list; -1: no LP lines
    std::vector<int32_t> allowed;    ///< token ids listed instead of the K most likely (the server renormalises them)
};

/// Stores `value` in `r` if `key` is logprobs or allowed; false for any other key.
bool read_logprob_key(const std::string& key, const std::string& value, LogprobRequest& r);

/// Formats the LP line of `token` from the `vocab` logits of the row it was picked after (before penalties and
/// temperature): "LP <logprob> [id:logprob ...]\n", listing r.allowed (but `token` and ids past the vocabulary) or
/// else the r.top most likely tokens; "LP nan\n" when `logits` is null or `token` lies outside the vocabulary.
std::string format_logprob_line(const float* logits, int64_t vocab, int32_t token, const LogprobRequest& r);

/// One conversation's LP lines in the paged engine: the model formats one for each token it picks (note), the engine
/// prints those of the tokens a step kept (take).  A step's rejected guesses leave lines that a later row at the same
/// position replaces.
class RowLogprobs {
public:
    /// Starts a request that asked for `r` and whose first answer token follows the row at `first`.
    void start(LogprobRequest r, int64_t first);
    /// Formats the LP line of `token`, picked after the row at `pos`, when the request asked for LP lines and the row
    /// is past its prompt; `copy` then writes that row's `vocab` head logits on the host (false: they are lost).
    void note(int64_t pos, int32_t token, int64_t vocab, const std::function<bool(float*)>& copy);
    /// Gets and forgets the LP line of the token picked after the row at `pos`: "" when the request asked for none,
    /// "LP nan\n" when no line was noted.
    std::string take(int64_t pos);

private:
    LogprobRequest request_;
    int64_t first_ = 0;
    std::map<int64_t, std::string> lines_;   ///< by the row's position
    std::vector<float> logits_;              ///< note's host copy of one row
};

/// Reads comma-separated hexadecimal numbers (the img_keys option: one key per picture); empty if any is malformed.
std::vector<uint64_t> parse_hex_list(const std::string& text);
/// Reads comma- or space-separated token ids from `s` into `out`; false (with `err`) for a malformed or empty list.
bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err);

}  // namespace strata::serve
