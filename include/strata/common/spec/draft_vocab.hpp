// include/strata/common/spec/draft_vocab.hpp - which draft-vocabulary file a DFlash2-alone start reads its draft head from.
//
// The subset is `--mtp-draft-vocab FILE`, else `<--mtp DIR>/draft_vocab.bin`.  A start without it would run the drafter's head over the whole vocabulary without a word
// (+0.7 ms a round measured on Metal), so a missing file is an error at start.  Header only, no device types.
#pragma once

#include <functional>
#include <string>

namespace strata::spec {

struct Df2Vocab {
    bool ok = false;
    std::string path;        ///< the file the drafter binds (the wanted path, even when it is missing)
    std::string why;         ///< ok == false: the line for stderr
};

inline Df2Vocab df2_vocab_resolve(const std::string& mtp_dir, const std::string& explicit_file,
                                  const std::function<bool(const std::string&)>& exists) {
    Df2Vocab r;
    const std::string wanted = explicit_file.empty() ? mtp_dir + "/draft_vocab.bin" : explicit_file;
    if (exists(wanted)) { r.ok = true; r.path = wanted; return r; }
    r.path = wanted;
    r.why = "--draft dflash2: the draft vocabulary " + wanted + " does not exist (--mtp DIR needs DIR/draft_vocab.bin, or --mtp-draft-vocab FILE); the drafter would run its head over the whole vocabulary";
    return r;
}

}  // namespace strata::spec
