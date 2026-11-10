// src/models/deepseekv41/ds4_options.cpp - the model's own command-line options (ds4_model.hpp: read_option), which
// strata-dsv41 and strata-ds41 both read, and the drafts' confidence cut both make.
#include "ds4_model.hpp"
#include "ds4_experts_rule.hpp"
#include "strata/common/spec/confidence.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::ds4 {

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    const char* p = s.c_str();
    while (*p) {
        char* e = nullptr;
        const long x = std::strtol(p, &e, 10);
        if (e == p) break;
        v.push_back((int) x);
        p = e;
        while (*p == ',' || *p == ' ') ++p;
    }
    return v;
}

bool read_option(const std::string& a, const std::function<std::string(const char*)>& val, Options& o,
                 bool& drafting) {
    if (a == "--pack") o.pack = val("--pack");
    else if (a == "--gpus") {
        for (int g : parse_ids(val("--gpus"))) o.gpus.push_back(g);
    } else if (a == "--max-context") o.max_context = std::atoll(val("--max-context").c_str());
    else if (a == "--reserve-mib") o.reserve_mib = std::atoll(val("--reserve-mib").c_str());
    else if (a == "--vram-experts") {
        o.vram_experts = std::atof(val("--vram-experts").c_str());
        o.vram_experts_auto = false;
    }
    else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val("--cpu-threads").c_str());
    else if (a == "--expert-profile") o.profile = val("--expert-profile");
    else if (a == "--profile-readonly") o.profile_readonly = true;   // (read the placement, never write the counts back)
    else if (a == "--ssd-cache") o.ssd_cache_gb = std::atof(val("--ssd-cache").c_str());
    else if (a == "--ssd-policy") {
        const std::string v = val("--ssd-policy");
        o.ssd_policy = v == "lfu" ? 1 : v == "lfulru" ? 2 : 0;
    }
    else if (a == "--no-expert-cache") o.expert_cache = false;
    else if (a == "--dspark") {
        const std::string v = val("--dspark");
        drafting = v != "off";
        o.dspark = v != "none";   // none: not even loaded
        o.dspark_auto = v == "auto";
    } else if (a == "--dspark-conf") o.dspark_conf = (float) std::atof(val("--dspark-conf").c_str());
    else if (a == "--experts") {
        const std::string v = val("--experts");
        o.experts = v == "auto" ? -1 : std::max(1, std::atoi(v.c_str()));
    }
    else if (a == "--dspark-block") o.dspark_block = std::max(0, std::atoi(val("--dspark-block").c_str()));
    else if (a == "--tp") o.tp = std::max(1, std::atoi(val("--tp").c_str()));
    else if (a == "--prefill-chunk") o.prefill_chunk = std::max(0, std::atoi(val("--prefill-chunk").c_str()));
    else if (a == "--threads") o.threads = val("--threads") != "off";
    else if (a == "--vision") o.vision = val("--vision") != "off";
    else if (a == "--mmap-experts") {
        const std::string v = val("--mmap-experts");
        o.mmap_experts = v == "on" ? 1 : v == "off" ? 0 : -1;
    } else
        return false;
    return true;
}

int Model::count_confident(const std::vector<float>& conf, int most, float conf_min, float cum_min) {
    return strata::spec::count_confident_drafts(conf.data(), std::min((int) conf.size(), most),
                                               conf_min, cum_min);
}

}  // namespace strata::ds4
