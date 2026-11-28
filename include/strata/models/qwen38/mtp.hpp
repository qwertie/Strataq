// include/strata/models/qwen38/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "strata/models/qwen38/layer.hpp"
#include "strata/base/gpu/pinned_ring.hpp"
#include "strata/base/gpu/shape_graph_cache.hpp"
#include "strata/models/qwen38/session.hpp"
#include "strata/common/kernels/sampler.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768, const MtpDrafter* shared = nullptr);
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    /// --mtp-hnorm stream (opt-in; before the first draft or prefill): pre_fc_norm_hidden normalizes each
    /// hyper-connection stream on its own, as llama.cpp's qwen4exp MTP graph does, instead of one RMS over all four.
    void set_hnorm_per_stream(bool on) { hnorm_stream_ = on; }
    bool hnorm_per_stream() const { return hnorm_stream_; }
    /// --mtp-q4 (opt-in; before load): the draft layer's Q8_0 projections and its draft head run from 4-bit (Q4_0)
    /// copies made at load/bind - fewer bytes per draft step; drafts only, the verify window decides every token.
    /// (The prompt path's batched K/V pass keeps the Q8_0 originals.)
    void set_q4(bool proj, bool head) { q4_ = proj; q4_head_ = head; }
    /// Before load: the layer's K/V lives in each conversation's page-pool pages (strata-q38, with
    /// session_set_pool_state), not in a buffer of its own; load carves none, and use_kv_units points it at a
    /// conversation's.
    void set_kv_in_pages(bool on) { kv_in_pages_ = on; }
    /// Where strata-q38's page pool keeps the layer's 512 routed experts (set_pool_experts).
    struct PoolExperts {
        const int32_t* res = nullptr;    ///< device: each expert's first page, or < 0 while it is not in VRAM
        const uint8_t* base = nullptr;   ///< the pool's page 0
        int64_t page_bytes = 0;
    };
    /// Before load (strata-q38): the 512 routed experts are the page pool's, where `*pool` says (filled by the
    /// caller before the first draft and kept as long as this drafter), not a block of this drafter's own, so load
    /// reads none.  A draft step's router then favors the experts in VRAM by the verify's route prior
    /// (get_route_prior), and a routed expert that is not in VRAM is left out as the verify's tail skip would leave it
    /// (tail_skip_rank); when it may not be, the chain ends before that step.
    void set_pool_experts(const PoolExperts* pool) { pool_ = pool; }
    const PoolExperts* get_pool_experts() const { return pool_; }
    /// With set_pool_experts: gets the expert ids the last draft's steps routed to, k per step, for the steps it ran
    /// (its drafts and the step that ended its chain).
    std::span<const int32_t> get_routed() const;
    /// With set_kv_in_pages: makes the next drafts and prompt passes read and write the K/V of the conversation whose
    /// units table (kv_units.hpp, one entry per block of the session's max_cells and 2 more) is `units`, by copying
    /// it on the drafter's stream into the table the drafter's graphs read.
    bool use_kv_units(const int64_t* units, std::string& err);
    /// --mtp-draft-vocab FILE (opt-in; before bind): the draft head's token subset from FILE instead of
    /// rt/draft_vocab.bin (e.g. data/draft_vocab_en.bin, 40K tokens)
    void set_draft_vocab(const std::string& path) { dvocab_path_ = path; }
    int max_t() const { return max_t_; }
    uint64_t vram_bytes() const { return vram_; }

    bool experts_resident() const { return experts_ != nullptr && !experts_lent_; }

    /// The draft layer's K/V state (read-only: --serve's STRATA_STATE_HASH check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// The VRAM bind() will allocate for a native head of `head_row_bytes` per vocabulary row: the draft logits and
    /// the draft head over rt/draft_vocab.bin's subset.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const;
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
              const MtpDrafter* shared = nullptr);

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.  `sync` false: returns without waiting
    /// for the drafter's stream (the caller orders on it: `stream()`).
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                 bool sync = true);
    cudaStream_t stream() const { return cs_; }

    /// Successful producer result, including a generated row that failed admission.
    /// budget is the actual producer max_steps; failure/entry resets both fields.
    /// With result supplied, n_drafts (if supplied too) is also reset on entry.
    struct DraftResult { int generated = 0; int budget = 0; };

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr, DraftResult* result = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    /// STRATA_SPEC_COST: called after each draft step with (index, probability) instead of the `min_p` test; false ends the chain.
    /// Also called once for the last step of a chain that reached its length (so the rule sees every draft). Empty: min_p decides.
    void set_chain_rule(std::function<bool(int, float)> f) { chain_rule_ = std::move(f); }

    void set_ple_session(SessionState* s) { ple_ss_ = s; }
    std::function<bool(int, float)> chain_rule_;
    bool coupled() const { return coupled_active_; }
    /// PROBABILISTIC DRAFT ACCEPTANCE (core/spec_prob.hpp, STRATA_SPEC_PROB=1): this request's drafts are SAMPLED from
    /// q and each draft j's distribution is in spec_q() + j * kSpecQStride (ids, -1 terminated; then float bits),
    /// valid on the host once draft() has returned.  False for greedy requests and where the mode is off.
    bool prob() const { return coupled_active_ && h_q_ != nullptr; }
    const int32_t* spec_q() const { return h_q_; }

    // ---- --pipeline-windows 2: the chain as one asynchronous launch (the round and its steps back to back, no host
    // wait), its first `n_force` steps fed the given tokens instead of their own picks (teacher forcing: the drafts of
    // a window already in flight), each output read once the event after its step has completed.
    /// Before `load`: the round/step graphs get the forcing kernel (a no-op while its token is -1), and the drafter's
    /// stream the highest priority (STRATA_MTP_PRIORITY=0: the default priority).
    void set_force_capture(bool on) { force_on_ = on; }
    /// The rows the next round reads (null: the bound `window_R`), copied into the bound buffer before the round.
    void set_source_R(const float* rows) { src_R_ = rows; }

    double ms_draft = 0, ms_prefill = 0;
    double ms_src = 0;      ///< the host time spent staging the source rows (a cross-card copy when the drafter is on another card)
    int64_t n_src = 0;
    int64_t rounds = 0;

    /// E-9: the prompt path computes this layer's prompt K/V in batches (Prefill::draft_kv): its tensors, its K/V
    /// state, the first cell a round can still read, its device, and a wait for its own stream.
    const float* tensor_f32(const char* name) const { return f32(name); }
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    QsaState& kv_state_rw() { return st_; }
    /// The first cell whose K/V the draft layer can still read: the window's start for this prompt, and never below
    /// the KVA floor (cells built from approximated rows are not read, so they are not built either).
    int64_t first_needed() const {
        const int64_t w = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
        return std::max<int64_t>(w, cell_floor_);
    }
    /// The cells the draft layer attends to (0 = every cell), as load() settled it.
    int64_t window() const { return window_; }
    /// KVA: call before load().  The draft layer's attention then reads its cell range through a floor, which stays 0
    /// (the plain window, cell for cell) until set_cell_floor() raises it.  Off (the default): the code is 0.2.6's.
    void enable_cell_floor(bool on) { floor_on_ = on; }
    int device() const { return device_; }
    bool idle(std::string& err) {
        if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
        return true;
    }

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    /// False (with `err`) while the experts are lent to a prompt: the prompt path's ring is pointed at their bytes
    /// (STRATA_MTP_ROOM_MIN), so a decode pass would read the prompt's staged blobs as experts.  Every decode entry
    /// checks this before it launches anything.
    bool decode_ready(std::string& err) const {
        if (experts_lent_) { err = "mtp: the 512 experts are lent to a prompt (STRATA_MTP_ROOM_MIN)"; return false; }
        return true;
    }
    /// The layer's front for T rows at step rows [row0, +T): the embedding, the fc projections, the attention
    /// hyper-connection read (R_, inj_, mixed_) and the K/V appended.
    bool record_front(int T, int row0, cudaStream_t cs, std::string& err);
    /// The rest for row 0 at step row `step_row`: the attention from the q projection, the MLP, the final mixer,
    /// the head, the draft and its probability.  Needs row 0's R_, inj_ and mixed_ (its q8_1 rows in xq_).
    bool record_rest(int step_row, cudaStream_t cs, std::string& err);
    void norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs);
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< E-4: without the mapped staging (inputs copied on device)
    bool capture_round(int T, bool coupled, std::string& err);
    bool capture_step(int j, bool coupled, std::string& err);
    /// Captures the missing graphs of a draft round over T rows and of its chain steps 1..steps-1, none of them
    /// freeing another for VRAM: the round launches them all.
    bool capture_draft(int T, int steps, bool coupled, std::string& err);
    // coupled draft sampling: its own round/step graphs (the argmax ones stay as they were), the request's
    // parameters and the penalty ring (mapped staging + device copies), the split scratch, token id -> subset index
    bool setup_coupled(std::string& err);
    bool coupled_ok_ = false, coupled_active_ = false;
    bool coupled_rec_ = false;   ///< record_forward: the full layer ends in the coupled sampler (draft coupled_j_)
    int coupled_j_ = 0;
    strata::kernels::SamplerParams *h_cparams_ = nullptr, *m_cparams_ = nullptr, *cparams_ = nullptr;
    int32_t *h_chist_ = nullptr, *m_chist_ = nullptr, *cring_ = nullptr, *dinv_ = nullptr;
    int32_t *h_q_ = nullptr, *m_q_ = nullptr;   ///< STRATA_SPEC_PROB: the drafts' q lists (mapped, max_t_ rows)
    void* cscratch_ = nullptr;
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    SessionState* ple_ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int max_drafts_ = 1 << 30;
    bool hnorm_stream_ = false;
    bool q4_ = false, q4_head_ = false;
    uint8_t* dense4_ = nullptr;              ///< --mtp-q4: Q4_0 copies of the Q8_0 tensors
    std::vector<std::pair<std::string, uint64_t>> q4_off_;
    int dhead_type_ = -1;                    ///< the draft head subset's ggml type (the main head's, or Q4_0)
    std::string dvocab_path_;
    std::string vocab_file() const { return dvocab_path_.empty() ? rt_dir_ + "/draft_vocab.bin" : dvocab_path_; }
    /// a projection's weights for the draft layer's own pass: the Q4_0 copy under --mtp-q4, else the Q8_0 original
    const void* wq(const char* name, int& type) const;
    bool make_q4_dense(const std::vector<uint8_t>& blob, std::string& err);
    bool make_q4_head(std::string& err);
    std::vector<float> lg_host_;
    /// Stages rows 0 .. T - 1 of a prompt chunk, round or chain: `tokens` at the cells from `cell0` on.
    void put_rows(const int32_t* tokens, int T, int64_t cell0);
    /// Stages the cell row `row` writes its K/V to (its step table entry and every head's position).
    void put_cell(int row, int64_t cell);
    // --pipeline-windows 2 (chain_launch)
    bool stage_source_R(int T, std::string& err);   ///< set_source_R's rows into the bound buffer
    bool force_on_ = false, chain_live_ = false;
    const float* src_R_ = nullptr;
    int32_t *h_force_ = nullptr, *m_force_ = nullptr;   ///< the forced tokens (mapped), -1 = the step's own pick
    cudaEvent_t ev_chain_ = nullptr;
    cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
    int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
    int32_t chain_tok_[8] = {};
    float chain_prob_[8] = {};
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    PinnedRing pf_ring_;          ///< #1996: the records above go up from a pinned slot, not from the pageable vector they are built in
    int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaStream_t side_ = nullptr;                    // the shared expert's branch of the draft graphs (CUDA)
    cudaEvent_t sh_fork_ = nullptr, sh_join_ = nullptr;
    /// What a captured graph runs: a prompt chunk (from the mapped staging or, Dev, from device inputs), a draft
    /// round, or one draft step of the chain; Coupled: with coupled draft sampling.
    enum GraphKind { kPrefillGraph, kPrefillDevGraph, kRoundGraph, kRoundCoupledGraph, kStepGraph, kStepCoupledGraph };
    /// The captured graphs by (kind, rows of the window or step index 1..8); short of VRAM, the least recently used
    /// are freed and captured again when next used.
    strata::core::ShapeGraphCache<std::pair<int, int>> graphs_;
    /// Gets the graph of `kind` for `n` (null when not captured) and marks it as just used.
    cudaGraphExec_t graph(GraphKind kind, int n) { return graphs_.find({kind, n}); }
    /// Ends the capture on cs_ and, when `ok` (the recording succeeded), keeps it as the graph of (`kind`, `n`).
    bool finish_capture(bool ok, GraphKind kind, int n, const char* what, std::string& err);

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    const PoolExperts* pool_ = nullptr;   ///< set_pool_experts: where the routed experts are (null: experts_)
    /// With pool_, mapped: PoolGroupArgs::chain's four ints, the route prior's bonus (a float), then from
    /// kPoolRoutedAt each step's routed ids
    int32_t *h_pool_ = nullptr, *m_pool_ = nullptr;
    static constexpr int kPoolRoutedAt = 8;
    // the prompt room: the experts' pinned host snapshot, its size, and whether a prompt has taken their contents
    // (the block itself stays where it is - see experts_lend)
    void* experts_host_ = nullptr;
    uint64_t experts_host_bytes_ = 0;
    bool experts_lent_ = false;
    // Slot drafters borrow immutable weights and head; each still owns its state and scratch.
    bool owns_weights_ = true;
    bool owns_draft_head_ = true;
    void* state_arena_ = nullptr;
    QsaState st_;
    bool kv_in_pages_ = false;
    int64_t* kv_units_ = nullptr;   ///< with set_kv_in_pages: the units table `st_` reads, filled by use_kv_units
    int64_t kv_blocks_ = 0;         ///< its entries
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t cell_floor_ = 0;    // KVA: no draft cell below this one is read or built (set_cell_floor)
    bool floor_on_ = false;     // enable_cell_floor: the attention goes through window_ids_floor
    int32_t *h_floor_ = nullptr, *m_floor_ = nullptr;   // the floor, mapped: the kernel reads it when it runs
    int64_t prompt_len_ = 0;
    const int32_t* rr_res_ = nullptr;
    float rr_margin_ = 0.0f;
    int rr_lo_ = 6, rr_hi_ = 9;
    float* probs_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' and row_top_prob_split's partials and counters
    uint8_t* top_scratch_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    /// gr_read's scratch of the drafter's own (not the session's ss.block.gr: the last stage's verifier uses that one,
    /// and with --pipeline-windows a chain and that stage's window can run on the card at once)
    strata::kernels::GrWorkspace own_gr_;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
