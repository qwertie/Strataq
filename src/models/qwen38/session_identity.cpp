// src/models/qwen38/session_identity.cpp - what a saved Qwen3.8 conversation is bound to (session_identity.hpp).
#include "strata/models/qwen38/session_identity.hpp"

#include "strata/common/state/conversation_file.hpp"

#include <cstdlib>
#include <vector>

namespace strata::qwen38 {

bool get_session_model_fingerprint(const Model& m, const strata::core::ExpertSource* experts, uint64_t& fingerprint,
                                   std::string& err, const std::function<void()>& next_file) {
    const Options& o = m.o;
    // what the loaders actually resolved: the CLI inputs, the pack's files and - from the expert source itself - every
    // file its experts were read from (native_experts.txt can name GGUFs per layer/role)
    strata::core::SessionInputs in;
    in.native_shards = o.native_shards;
    in.native_dense = o.native_dense_gguf;
    in.native_head = o.native_head_shards;
    in.embedding = o.embd_gguf;
    in.ple = o.ple_gguf;
    in.pack = o.pack;
    in.mtp = o.mtp.empty() ? std::string() : o.mtp + "/dense.bin";
    if (experts != nullptr)
        in.experts = experts->model_inputs();
    std::vector<strata::core::SessionModelFile> files = strata::core::session_model_inputs(in);
    // the pack's other files the loader reads (weights.cpp, expert_layout.cpp)
    if (!o.pack.empty()) {
        for (const char* n : {"index.txt", "dense.bin"})
            files.push_back({std::string("pack ") + n, o.pack + "/" + n});
        for (const char* n : {"embd.bin", "extra.bin", "manifest.json"})
            files.push_back({std::string("pack ") + n, o.pack + "/" + n, true});
        if (experts == nullptr)
            files.push_back({"pack experts.bin", o.pack + "/experts.bin", true});
    }
    if (!o.mtp.empty()) {
        for (const char* n : {"dense.txt", "experts.bin"})
            files.push_back({std::string("mtp ") + n, o.mtp + "/" + n});
        files.push_back({"mtp draft_vocab.bin", o.mtp + "/draft_vocab.bin", true});
    }
    // each file is one blocking read of at most 2 MiB (head and tail): announced before it starts
    if (next_file)
        next_file();
    return strata::core::session_model_fingerprint(files, fingerprint, err, next_file);
}

uint64_t get_session_config_fingerprint(const Model& m) {
    const Options& o = m.o;
    strata::core::SessionConfig c;
    c.engine_version = STRATA_VERSION;
#if defined(STRATA_USE_HIP)
    c.backend = "hip";
#else
    c.backend = "cuda";
#endif
    c.kv = o.kv;
    c.kv_resident = o.kv_resident;
    c.mtp_window = o.mtp.empty() ? -1 : o.mtp_window;
    const char* rot = std::getenv("STRATA_KV_ROT");
    c.kv_rot = rot != nullptr && rot[0] == '1';
    const strata::kernels::RopeScaling& rope = m.rope_cfg;
    c.rope = {(int64_t) rope.type, rope.freq_base, rope.factor, rope.freq_scale(), rope.orig_ctx,
              rope.ext_factor, rope.attn_factor, rope.beta_fast, rope.beta_slow};
    c.cvec = m.cvec_digest;
    c.switches = {
        {"no_ple", o.no_ple}, {"native_bf16", o.native_bf16}, {"native_bf16_extra", o.native_bf16_extra},
        {"native_ple_key", o.native_ple_key}, {"native_moe_combine", o.native_moe_combine},
        {"native_gdn", o.native_gdn}, {"native_flash_attn_short", o.native_flash_attn_short},
        {"native_qsa_indexer", o.native_qsa_indexer}, {"native_qsa", o.native_qsa},
        {"native_rope", o.native_rope}, {"native_ple_postops", o.native_ple_postops},
        {"native_router", o.native_router}, {"cpu_oracle_q8_0", o.cpu_oracle_q8_0},
        {"gr_fp32_activations", o.gr_fp32_activations}, {"gr_native_mmvf", o.gr_native_mmvf},
        {"native_preset", !o.native_preset.empty()}, {"shared_late", o.shared_late},
        {"keep_canonical", o.keep_canonical}, {"no_fused_gr", o.no_fused_gr},
        {"no_fast_attn", o.no_fast_attn}, {"no_fused_gdn", o.no_fused_gdn},
        {"no_fast_select", o.no_fast_select}, {"vision", o.vision}};
    return strata::core::session_config_fingerprint(c);
}

}  // namespace strata::qwen38
