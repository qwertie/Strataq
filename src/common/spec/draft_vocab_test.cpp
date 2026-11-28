// draft_vocab_test: the DFlash2-alone start's draft-vocabulary choice (include/strata/common/spec/draft_vocab.hpp).
#include "strata/common/spec/draft_vocab.hpp"

#include <cstdio>
#include <set>
#include <string>

using strata::spec::df2_vocab_resolve;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++g_fail; } } while (0)

int main() {
    const std::set<std::string> files = {"/mtp/with/draft_vocab.bin", "/given.bin"};
    auto exists = [&](const std::string& p) { return files.count(p) > 0; };

    // the MTP folder has the subset: it is used
    { auto r = df2_vocab_resolve("/mtp/with", "", exists); CHECK(r.ok && r.path == "/mtp/with/draft_vocab.bin"); }
    // a file the person named wins over the folder's
    { auto r = df2_vocab_resolve("/mtp/with", "/given.bin", exists); CHECK(r.ok && r.path == "/given.bin"); }
    // the MTP folder has none: a missing file is an error
    { auto r = df2_vocab_resolve("/mtp/empty", "", exists); CHECK(!r.ok && r.why.find("/mtp/empty/draft_vocab.bin") != std::string::npos); }
    // a file the person named that does not exist is their mistake: no silent fallback to the folder's
    { auto r = df2_vocab_resolve("/mtp/with", "/typo.bin", exists); CHECK(!r.ok && r.why.find("/typo.bin") != std::string::npos); }

    if (g_fail == 0) std::printf("draft_vocab_test: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
