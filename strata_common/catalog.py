"""The model catalog: every model family and size Strata offers, the files they are made of and the pins that check
those files (the Hugging Face revision, the size and the SHA-256 of each).

Data, the two URL helpers it is built with and the DeepSeek layout helpers, standard library only: setup.py (which runs
before any package is installed), the Terminal, Control and the MCP server read the same tables from here.  The
download, install and fit logic that uses them is setup.py's.  The download folders in FAMILIES and the other tables
are built when this module is first imported, from HF_ENDPOINT as it is then.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# #214: every Hugging Face file comes from a fixed commit of its repository (the `sha` of
# https://huggingface.co/api/models/<repo> when this was pinned), so a checkout installs the same files on any
# day.  Every file has a fixed size and SHA-256 too (REPO_PINS); a revision the repository no longer has stops the
# download (download()), unless a tester sets STRATA_ALLOW_UNPINNED=1.
HF_REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",        # 2026-09-29
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d",   # 2026-10-07 (adds IQ3_S)
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",  # 2026-09-29
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",                   # 2026-09-30
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",                  # 2026-10-12
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": "1ada08422e404aca51ab9df665f089f34c419c6b",           # 2026-10-13
    "unsloth/DeepSeek-V4.1-Flash-1017-GGUF": "8a6ec9cf118e785a362ea94ee123c61c70553be6",             # 2026-10-30 (adds UD-IQ1_S / UD-IQ1_M; every file pinned from the 2026-10-20 commit has the same size and SHA-256 here)
    "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF": "65d53bc8be65123e255ff02f3ded958b6dbf54fd",    # 2026-10-28 (6 shards)
}


HF_DEFAULT = "https://huggingface.co"


def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co.  The pinned revisions and the SHA-256 checks are the same whichever host serves the files."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf(repo: str) -> str:
    """The download folder of a Hugging Face repository at its pinned revision."""
    return f"{hf_endpoint()}/{repo}/resolve/{HF_REVISIONS[repo]}/"


HF = hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")

# NVFP4 experts (MODEL-UPDATE Phase 4; the engine path is ported from sergqwer/strata-nvfp4, MIT).  Ready-made Strata
# packs on Hugging Face by the fork's author: nothing is converted here.  Every file is pinned by its upload's commit,
# size and SHA-256 (the fork's setup.py NVFP4_REPOS, checked against the Hub's LFS pointers 2026-10-12).  The FP8 n-gram
# table and the BF16 token embedding are the same files in both repositories (a second install links them).  The
# MTP draft layer is setup's own, as for every other model, so the repositories' mtp/ folder is not downloaded.
# role: pack = the experts pack folder (index.txt, experts.bin, ...), dense = the dense GGUF (--native and
# --native-dense-gguf), ple = the FP8 n-gram table (--ple-gguf), embd = the BF16 token embedding (--embd-gguf)
NVFP4_GPTQ_REPO = "Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata"     # Qwen's BF16 experts by GPTQ (the fork)
NVFP4_NVIDIA_REPO = "Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata"  # nvidia/Qwen3.8-Flash-Next-NVFP4, converted
_NVFP4_TOKENIZER = {
    "pack/tokenizer/chat_template.jinja": ("pack", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
    "pack/tokenizer/merges.txt": ("pack", 3600844, "080cf95432173729d160346b7882bd2a644b07f7a3813a3eb73f4e66787bfde9"),
    "pack/tokenizer/token_type.json": ("pack", 744960, "5088c8c298fc06af8382ddb3b76c888703ac2634e82263b97eedcc2ad202738b"),
    "pack/tokenizer/tokenizer.json": ("pack", 554, "87be2ac47d8bc7393b7435df6ccd276740019a393941a2585d5edce59f450c54"),
    "pack/tokenizer/vocab.json": ("pack", 5737005, "4ba64f0332abcfb0b600b7df1537d1e836e68009d5fb7cdb77339738dc6365c4"),
}
_NVFP4_SHARED = {
    "ple-fp8.gguf": ("ple", 51200246176, "37f89b8ce5a3425932763257c40229ec02723ad2ab14b4bbfc26a486f4725c9c"),
    "token-embd-bf16.gguf": ("embd", 1271398656, "386203af9070c7b43acb24ded8922876721d97bbad1a709b25cb49d694a7f51d"),
}
NVFP4_REPOS = {
    NVFP4_GPTQ_REPO: {"revision": "640a1db97446ddd5e89d2b70ecd0297987192c93", "files": {
        "pack/dense.bin": ("pack", 1538035200, "974df6dcefcae85e3fd2c19fcc12eeaacaadde8261ab732d5af5af6ce2e25a7a"),
        "pack/experts.bin": ("pack", 67948118016, "325e5e0d85c262a8d8278daf73b20696ec877d9a3ff584e57391fa69ad10bc64"),
        "pack/index.txt": ("pack", 93663, "f62dc71e1c74a72794db54d3917bc71131cc6eae6f3dcdc3fdada48ff70f93ab"),
        "pack/native_experts.txt": ("pack", 1599, "615f655fb28bf60b243664860b135710bf4177b111c6be89073e780a35de1018"),
        **_NVFP4_TOKENIZER,
        "qwen-nvfp4-gptq-dense.gguf": ("dense", 5991569472, "abdf74336b53d8f7ca3d99947f1eac8f19b8a544966750127cb61e561b127e3b"),
        **_NVFP4_SHARED}},
    NVFP4_NVIDIA_REPO: {"revision": "cca8f7fee5ce0af932097340a75e06df25c4e57d", "files": {
        "pack/dense.bin": ("pack", 1538625024, "1f1c2422069de292673ae6cee7d69f7090732dec69f3f125612a0de456279714"),
        "pack/experts.bin": ("pack", 67948118016, "bb42080bb7585505b64d091d5a734c05376bbd6171406fa9688a46fbc261b957"),
        "pack/index.txt": ("pack", 117490, "17793d1c5b187a73d3c1dc1677af4d4c9b23845394dd7e88bfa53396b41b77e3"),
        "pack/native_experts.txt": ("pack", 1590, "d7f8b4372bb122f3d711030e5d522bff1c8746ac7d133a59a2cc7277283e59b6"),
        **_NVFP4_TOKENIZER,
        "qwen-nvidia-nvfp4-dense.gguf": ("dense", 5992176896, "0f795ca0cbb908e0cf2b894b30103e21455b5a05346a7ef79da208540cfd2be3"),
        **_NVFP4_SHARED}},
}
NVFP4_FP4LOCAL_REPO = "nvidia/Qwen3.8-Flash-Next-FP4-Local"   # NVIDIA's QAD checkpoint (ModelOpt safetensors), prepared here
# Not a ready-made pack: the safetensors (95 GiB) are downloaded as NVIDIA ships them and tools/nvfp4_local_pack.py prepare
# turns them into the pack, the dense GGUF, the n-gram table and the draft layer on this PC (nothing of NVIDIA's weights is
# redistributed).  Role "src" = a file of the checkpoint; the prepared files are in FP4LOCAL_PREPARED.
NVFP4_REPOS[NVFP4_FP4LOCAL_REPO] = {"revision": "b9ce18a6887e18c089fdea8509a33e63429a3300", "files": {
        "config.json": ("src", 75217, "b50dbc52a0c83b0cd2a44a372c1fb6b5e0e6827d22f2611c97d4d030f92f6b28"),
        "generation_config.json": ("src", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"),
        "chat_template.jinja": ("src", 8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"),
        "merges.txt": ("src", 3353259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
        "tokenizer.json": ("src", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"),
        "tokenizer_config.json": ("src", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"),
        "vocab.json": ("src", 6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
        "model.safetensors.index.json": ("src", 47280974, "5fa166bbb1cf25adda455117563efa72a2072272001f459eedece131ffc913e1"),
        "tensors/model-00001-of-00021.safetensors": ("src", 5000643504, "ebf3648c5f4cd1f24e79789e7803498dc4e74cb52515cc515b2d22cf8161953b"),
        "tensors/model-00002-of-00021.safetensors": ("src", 5002792096, "79a643fc5c024089cecd32ca27dc43e671cbec70ff3628b728909b7656ac43ec"),
        "tensors/model-00003-of-00021.safetensors": ("src", 5002927400, "c45f942203db4c7f5a1071700b93b908495cc790271acddfff39091c2b953923"),
        "tensors/model-00004-of-00021.safetensors": ("src", 5003194672, "071ab9cf963ca0f1a6a194b4cec485f5de18ef9804da915b2e6abd0eb0c707af"),
        "tensors/model-00005-of-00021.safetensors": ("src", 5003151448, "f80d17870eb1d36917257494770984953476583c57737f61fdbd341c957b9715"),
        "tensors/model-00006-of-00021.safetensors": ("src", 5002892148, "3e9183d4ebb267fa92429184d5132d302cb1a98fd049149e5d01dccd4ca65795"),
        "tensors/model-00007-of-00021.safetensors": ("src", 5002907708, "aa1ad959c4d62e0bc4600fd0b57cc2939b463e9fb53dc01020e188552b52ce5a"),
        "tensors/model-00008-of-00021.safetensors": ("src", 5003103756, "0b18844027be74b84b240f23bf3a744bc29b35de0bd384f41c40cd85e6339326"),
        "tensors/model-00009-of-00021.safetensors": ("src", 5002976316, "2d0b6c66790c17f8b1fdcf442f541778c8f0a381aca26c11b9d7cfcf7c96f2f0"),
        "tensors/model-00010-of-00021.safetensors": ("src", 5002834136, "3ed9bd73745aebf6dc136f437b67d6b9fc3d38d25ad84300b399164bc87a72b7"),
        "tensors/model-00011-of-00021.safetensors": ("src", 5002685828, "4936ff28c6cc8b66653ddb9cae28bc284569a1a2f7116819a017dc1a0e517ed8"),
        "tensors/model-00012-of-00021.safetensors": ("src", 5002912920, "dd7df5a6ff73e9960412243f744eab9ec35f4be40da5bb158dc4ba8796b33742"),
        "tensors/model-00013-of-00021.safetensors": ("src", 5003237060, "28a455c8ea4ebaa847fb4cbed2bc1de9eff0b501dd3f848a3a3cfc38778e8500"),
        "tensors/model-00014-of-00021.safetensors": ("src", 4845006312, "6941676e685abf9817d9c48a17fdeda20b2b72d16217555252aa4c602b7f98b8"),
        "tensors/model-00015-of-00021.safetensors": ("src", 4860574920, "2a1129b5acfc24a45bf402aa63dcef08b531aedfdaf5233c65fd3d12dc303fdd"),
        "tensors/model-00016-of-00021.safetensors": ("src", 4950031028, "635e97fbad78fa988647d407e2b58e55d551ef94d798c9b6fb5550de25e86429"),
        "tensors/model-00017-of-00021.safetensors": ("src", 4950030896, "bf60d564d89a805b67f8be5f2847a1b1567a70dbc23d19754395c127568474ad"),
        "tensors/model-00018-of-00021.safetensors": ("src", 4950030896, "c5bde22f9364e82eb25973456795592fdcb4b158bc2095898979a2566bf4c562"),
        "tensors/model-00019-of-00021.safetensors": ("src", 4950030904, "da3dc3b4f428f5b9dd53362691043d0242f48d03adca9b7b553bd9f3a1e96350"),
        "tensors/model-00020-of-00021.safetensors": ("src", 4950030904, "4978bb6b8480cfc02d3c8bc84ba86202889f5f10d090bb17c21fee663cea7683"),
        "tensors/model-00021-of-00021.safetensors": ("src", 2700016864, "48203838f2c76921bddbf1cc51d20a23858c128821a21add55fa1778e5e25283"),
}}
FP4LOCAL_PREPARED = "prepared"                       # <models>/nvidia--Qwen3.8-Flash-Next-FP4-Local/prepared/

# the first release engine with NVFP4 experts (a source build of this checkout has them whatever its number says)
NVFP4_ENGINE = (0, 1, 44)
# the first engine that reads NVIDIA's NVFP4 n-gram table (strata.ple.format = nvfp4x16); a source build of this checkout has it
FP4LOCAL_ENGINE = (0, 1, 45)

# agentionai's Gyro files (Qwen3.8-Flash-Next with trellis-coded, Hadamard-rotated experts, TQ2_T / TQK6 / TQK7): one
# GGUF per size, the n-gram table inside it (hence the 58 / 92 GB downloads for 27.6 / 35 GB of memory).  The engine
# reads them from 0.1.45.1 on: the experts are native_experts.txt v5 (Hadamard-folded, hadamard.txt), which the engine reads since the
# APR port (commit f47b0385f, 2026-10-02, first in the 0.2.0 line).  The released 0.1.45 engine reads only up to v4 and refuses a
# Gyro pack, so 0.1.45 is NOT enough: 0.1.45.1 is the first engine package built after the reader was merged (checked: the
# "native_experts.txt v5 declares Hadamard-folded experts" message is in every 0.1.45.1+ strata binary and in no 0.1.45 release one).
# (ported from agentionai/Strata rc2, MIT); the sizes and SHA-256 are the Hub's LFS pointers
# of the revision pinned in HF_REVISIONS.  The model's license is Qwen's.
GYRO_ENGINE = (0, 1, 45, 1)
GYRO_FILES = {
    "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf":
        (58487015808, "c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87"),
    "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf":
        (91965774272, "7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f"),
}

# ISTA-DASLab's Qwen3.8-Flash-Next shards (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and the picture reader at the revision pinned in HF_REVISIONS: name -> (bytes, sha256),
# the Hub's LFS pointers (tools/hf_pin_fetch.py).  The n-gram table (shard 2) is the same file for all four sizes.
ISTA_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39225954592, "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
    "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}
# Swift 1.5's shards (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and its picture reader, same source and form.
SWIFT_FILES = {
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf":
        (39788473344, "ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf":
        (28363693824, "e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf":
        (44900816736, "46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf":
        (38842920384, "70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
        (39785790560, "3bddaa667c750f63baca766df9433e1afd35a139f401c5ca5e7ff560ceff3d23"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
        (36180282560, "b0b15f782af71eb471909d2f3313b2927c324962f86c084325c247cd411a0160"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf":
        (39799117984, "79e2a3873f1b05df835a1a191b20f2e74b9e12df949cd89861d17d98b6ef9678"),
    "Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf":
        (26750834816, "ef3bb04fb4e14f04624abd8acf25faf1603594a034050b048df85ec152ad5ae0"),
    "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf":
        (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
}
# The Coder's IQ1_M shards and its picture reader (the original's file), same source and form.
CODER_FILES = {
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf":
        (29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf":
        (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf":
        (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
}

# Round 0214: Unsloth re-published the repository on 2026-11-04 (commit d9620119, "Dynamic v3.1").  The files of UD-Q4_K_XL, UD-IQ4_XS (three shards),
# UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 stay at the commit in HF_REVISIONS (38bb39ee, pinned since 0.1.31); the sets that are new there (UNSLOTH_V31_SHARDS) name this
# commit in their MODELS entry ("revision"), so a repository has more than one revision and the pin of a file is found by the commit its URL names (file_pin).
UNSLOTH_REPO = "unsloth/Qwen3.8-Flash-Next-GGUF"
UNSLOTH_V31_REVISION = "d96201194f103c906994f283f504209dc8d2a10c"          # 2026-11-04
UNSLOTH_V31_IQ4_XS = "UD-IQ4_XS-v31"      # the catalog name of the new UD-IQ4_XS (its files on the Hub have the pinned one's names, with four shards instead of three)
UNSLOTH_V31_ENGINE = (0, 1, 57)       # the engine that ships with these sets (i-quant experts with Q2_0 / IQ4_NL / Q5_0 downs, a Q6_K dense side)

MODELS = {
    # the original model only for now: Swift 1.5's Q2_0 files split one layer's experts across the two shards, which
    # the pack tool (tools/iq_pack.py) cannot prepare yet (#171)
    "Q2_0": {"about": "2-bit, the fastest", "download_gb": 66.4, "ram_gb": 48, "arena_gb": 34.0, "families": ("qwen",)},
    "IQ2_XS": {"about": "2-bit i-quant, a little better quality, close in speed", "download_gb": 68.0, "ram_gb": 48,
               "arena_gb": 35.5},
    "IQ3_XXS": {"about": "3-bit i-quant, better quality, slower (more CPU work per token)", "download_gb": 75.8,
                "ram_gb": 60, "arena_gb": 42.9},
    # matches the full BF16 model on the published benchmarks; Swift 1.5 got an IQ3_S tier of its own (#1651)
    "IQ3_S": {"about": "3.5-bit i-quant, the best quality (matches the full model), the slowest; needs a 64 GB PC "
                       "with little else running", "download_gb": 83.6, "ram_gb": 62, "arena_gb": 50.3},
    # the Coder release: 256 of the 512 experts kept (the ones code, tools and vision use), IQ2_S-IQ4_XS like IQ3_S
    "IQ1_M": {"about": "the Coder's only size: half the experts, stored like IQ3_S (3.5 bits)", "download_gb": 58.4,
              "ram_gb": 32, "arena_gb": 23.4, "families": ("coder",)},
    # #621: Unsloth's UD-IQ4_XS - IQ3_S gate/up experts with IQ4_NL (43 layers) or Q8_0 (5) downs, the dense side as
    # UD-Q4_K_XL's; three shards.  A regular choice from 0.1.39 (no longer experimental).  Its 59.5 GB of experts: a
    # RAM budget of them, like UD-Q4_K_XL, but far fewer read from the SSD on a 64 GB PC and none from ~80 GB of RAM.
    # Images: the vision path has no restriction for this pack (the same base model and image encoder), so setup asks
    "UD-IQ4_XS": {"about": "~4-bit i-quant (Unsloth Dynamic), between IQ3_S and UD-Q4_K_XL in quality; on a PC with "
                           "less than ~80 GB of RAM part of its experts are read from the SSD",
                  "download_gb": 93.7, "ram_gb": 48, "arena_gb": 59.5, "families": ("unsloth",), "budget": True,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "engine": (0, 1, 38),
                  "vision": True},
    # Unsloth's 4-bit file (docs/UNSLOTH_Q4.md); experimental until MODEL-UPDATE Phase 2.  Its 77 GB of experts stay in
    # RAM from ~94 GB of RAM (unsloth_budget); on a 64 GB PC the engine keeps a RAM budget of them
    # (--resident-budget-gib, chosen below) and reads the rest from the GGUF on the SSD.  It runs on AMD too (R9700:
    # the quality yardstick and the decode/prompt numbers in docs/QUANTS.md), so it is no longer NVIDIA-only (#429).
    "UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic): K-quant experts, between UD-IQ4_XS and UD-Q5_K_XL in quality",
                   "download_gb": 111.3, "ram_gb": 48, "arena_gb": 77.0, "families": ("unsloth",), "budget": True,
                   # #967 / MODEL-UPDATE: images run end to end with this file (CUDA: RTX A4000, GPU encoder; HIP:
                   # R9700, CPU encoder; the test picture's text and shapes as the IQ3_XXS pack answers them)
                   "vision": True},
    # MODEL-UPDATE: Unsloth's larger files (docs/QUANTS.md).  The same GGUF-in-place mode: every expert in RAM where it
    # fits with UNSLOTH_RESIDENT_BESIDE_GB more (unsloth_budget), else a RAM budget of them and the rest read from the
    # SSD while it answers - so "ram_gb" is the all-in-RAM need, and below it setup asks first (default no).  Their
    # shards 2 (the head) and 3 (the 54 GB Q8_0 PLE table) are the same bytes in all three files.  UD-Q5_K_XL and
    # UD-Q6_K_XL need the Q6_K expert kernels every engine has from MODEL-UPDATE Phase 1 on (Q5: layer 2's gate/up).
    "UD-Q5_K_XL": {"about": "5-bit (Unsloth Dynamic): Q5_K experts with Q8_0 downs, close to Q8_0 in quality",
                   "download_gb": 158.3, "ram_gb": 108, "arena_gb": 98.4, "families": ("unsloth",), "budget": True,
                   "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "engine": (0, 1, 44),
                   "vision": True},
    "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic): Q6_K experts with Q8_0 downs",
                   "download_gb": 169.2, "ram_gb": 118, "arena_gb": 109.2, "families": ("unsloth",), "budget": True,
                   "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "engine": (0, 1, 44),
                   "vision": True},
    "Q8_0": {"about": "8-bit, the reference quality (every expert Q8_0); for a 192 GB PC",
             "download_gb": 188.2, "ram_gb": 136, "arena_gb": 128.3, "families": ("unsloth",), "budget": True,
             "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf", "engine": (0, 1, 38),
             "vision": True},
    # Round 0214: Unsloth's v3.1 sets (revision UNSLOTH_V31_REVISION, pins UNSLOTH_V31_SHARDS): i-quant / Q2_0 experts, the last shard is the n-gram table
    # (IQ4_NL, 28.8 GB, in a shard of its own); "quant" is the name on the Hub (folder and file names).  The about lines come with the wording commit.
    # download_gb / arena_gb: the pinned shard sizes and the routed experts' bytes (tests/fixtures/unsloth_v31); ram_gb: the RAM-budget floor of UD-Q4_K_XL.
    "UD-IQ1_M": {"about": "~1-bit (Unsloth Dynamic v3.1), the smallest download", "download_gb": 68.0, "ram_gb": 48, "arena_gb": 34.8, "families": ("unsloth",), "budget": True,
                 "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "quant": "UD-IQ1_M",
                 "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    "UD-Q2_0": {"about": "2-bit (Unsloth Dynamic v3.1)", "download_gb": 70.5, "ram_gb": 48, "arena_gb": 37.4, "families": ("unsloth",), "budget": True,
                "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "quant": "UD-Q2_0",
                "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    "UD-IQ2_XS": {"about": "2-bit i-quant (Unsloth Dynamic v3.1)", "download_gb": 75.8, "ram_gb": 48, "arena_gb": 42.3, "families": ("unsloth",), "budget": True,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf", "quant": "UD-IQ2_XS",
                  "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    "UD-Q2_K_XL": {"about": "2-bit (Unsloth Dynamic v3.1) with some 3-bit experts", "download_gb": 79.6, "ram_gb": 48, "arena_gb": 46.0, "families": ("unsloth",), "budget": True,
                   "shards": 4, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "quant": "UD-Q2_K_XL",
                   "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    "UD-IQ3_XXS": {"about": "3-bit i-quant (Unsloth Dynamic v3.1)", "download_gb": 83.6, "ram_gb": 48, "arena_gb": 50.3, "families": ("unsloth",), "budget": True,
                   "shards": 4, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "quant": "UD-IQ3_XXS",
                   "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    "UD-Q3_K_XL": {"about": "3-bit (Unsloth Dynamic v3.1) with some 4-bit experts", "download_gb": 89.5, "ram_gb": 48, "arena_gb": 56.2, "families": ("unsloth",), "budget": True,
                   "shards": 4, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "quant": "UD-Q3_K_XL",
                   "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    # the v3.1 UD-IQ4_XS is another file from the pinned UD-IQ4_XS (345 of 1224 tensors changed type; four shards): its own name
    UNSLOTH_V31_IQ4_XS: {"about": "~4-bit i-quant (Unsloth Dynamic v3.1), a newer file than UD-IQ4_XS with other tensor types", "download_gb": 93.7, "ram_gb": 48, "arena_gb": 60.4, "families": ("unsloth",), "budget": True,
                         "shards": 4, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "quant": "UD-IQ4_XS",
                         "revision": UNSLOTH_V31_REVISION, "engine": UNSLOTH_V31_ENGINE, "vision": True, "uma_pool_auto": True},
    # MODEL-UPDATE Phase 4: NVFP4 experts, ready-made packs (NVFP4_REPOS).  63.3 GiB (68.0 GB) of experts in the
    # pack's experts.bin; the low-RAM mode reads them from there on a PC whose RAM does not hold them.  NVIDIA only
    # (the HIP and SYCL engines have no NVFP4 kernels).  Images: both packs keep Qwen's vision tower unquantized, so the
    # original's image encoder (ISTA's mmproj, the same file as for every Flash-Next model) reads pictures for them;
    # checked end to end on an RTX 5070 and an RTX 3060 against IQ3_XXS's answers on the same picture.
    # Gyro (agentionai): "ram_gb" is the memory the weights need in total, VRAM and RAM together ("mem_total"): Gyro-S
    # 27.6 GB, Gyro-M 35 GB (the rest of the download is the n-gram table, read from the SSD).  Neither is anyone's
    # default: setup and the MCP server mention them where the other sizes do not fit.  Images: the same mmproj as every
    # Flash-Next model; strata-vision reads only the vocabulary and header of the Gyro GGUF (tools/vision/gguf_meta.h),
    # whose TQK6 / TQK7 tensors llama.cpp does not know.
    "Gyro-S": {"about": "the smallest option, for PCs with little memory (27.6 GB of VRAM + RAM together); lower quality "
                        "than the 2-3-bit sizes, and the 2-bit Q2_0 decodes faster",
               "download_gb": 58.5, "ram_gb": 27.6, "arena_gb": 25.8, "families": ("gyro",), "mem_total": True,
               "file": "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf", "shards": 1, "engine": GYRO_ENGINE, "vision": True},
    "Gyro-M": {"about": "about IQ3_XXS quality with about 20% less GPU memory (35 GB of VRAM + RAM together); decodes "
                        "about 16-21% slower than Q2_0 on 12 GB cards",
               "download_gb": 92.0, "ram_gb": 35.0, "arena_gb": 32.1, "families": ("gyro",), "mem_total": True,
               "file": "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf", "shards": 1, "engine": GYRO_ENGINE, "vision": True},
    "NVFP4": {"about": "best quality of the NVFP4 versions (GPTQ, sergqwer's strata-nvfp4 pack); RTX 50 has FP4 tensor cores, "
                       "other NVIDIA cards run it at 8-bit precision", "download_gb": 128.0, "ram_gb": 80, "arena_gb": 68.0,
              "families": ("nvfp4",), "nvidia_only": True, "vision": True, "repo": NVFP4_GPTQ_REPO,
              "engine": NVFP4_ENGINE,
              "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's): https://huggingface.co/Qwen/Qwen3.8-Flash-Next"},
    "NVFP4-NVIDIA": {"about": "NVIDIA's own NVFP4 checkpoint (plain rounding), converted by sergqwer; the same size "
                              "and speed as NVFP4", "download_gb": 128.0, "ram_gb": 80, "arena_gb": 68.0,
                     "families": ("nvfp4",), "nvidia_only": True, "vision": True, "repo": NVFP4_NVIDIA_REPO,
                     "engine": NVFP4_ENGINE,
                     "license": "NVIDIA Open Model License (Licensed by NVIDIA Corporation under the NVIDIA Open Model "
                                "License) with Qwen3.8-Flash-Next's Qwen Community License 1.0: "
                                "https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4"},
    "NVFP4-QAD": {"about": "NVIDIA's FP4, trained for FP4 (NVIDIA's Qwen3.8-Flash-Next-FP4-Local) - slightly faster on short prompts; "
                           "a 102 GB download that this PC turns into a Strata pack once (about 20-40 minutes, 110 GB more disk); "
                           "RTX 50 cards have FP4 tensor cores, other NVIDIA cards run it at 8-bit precision. It reads pictures (the shared Qwen3.8-Flash-Next encoder)",
                  "download_gb": 102.2, "prepare_gb": 110.0, "ram_gb": 80, "arena_gb": 68.0,
                  "families": ("nvfp4",), "nvidia_only": True, "vision": True, "repo": NVFP4_FP4LOCAL_REPO,
                  "prepare": "fp4-local", "engine": FP4LOCAL_ENGINE,
                  "license": "NVIDIA Open Model Agreement (Licensed by NVIDIA Corporation under the NVIDIA Open Model "
                             "Agreement) with Qwen3.8-Flash-Next's Qwen Community License 1.0: "
                             "https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local"},
}
# The experimental Unsloth file's four shards at the pinned revision: name -> (bytes, sha256), checked after the
# download (setup trusts no other model file by name and size alone either: check_shards reads their directories).
UNSLOTH_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
}
# #621: UD-IQ4_XS's three shards at the same revision (sizes and SHA-256: the Hub's LFS pointers)
UNSLOTH_IQ4_XS_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf":
        (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf":
        (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf":
        (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
}
# MODEL-UPDATE: UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 at the same revision (sizes and SHA-256: the Hub's LFS pointers; shards
# 2 and 3 are byte-identical across the three)
UNSLOTH_BIG_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf":
        (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf":
        (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf":
        (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
    "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf":
        (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf":
        (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf":
        (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf":
        (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
    "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf":
        (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
    "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf":
        (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
    "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf":
        (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
    "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf":
        (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
    "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf":
        (49446841216, "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a"),
    "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf":
        (49668930400, "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97"),
    "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf":
        (34015618784, "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5"),
}
# Round 0214: the sets that are new at UNSLOTH_V31_REVISION (UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL and the v3.1 UD-IQ4_XS), name -> (bytes,
# sha256): the Hub's LFS pointers (phaseA-tests/quants0211/unsloth-head-pins.json).  The last shard of each is the n-gram table.  No name is in UNSLOTH_SHARDS,
# UNSLOTH_IQ4_XS_SHARDS or UNSLOTH_BIG_SHARDS (the v3.1 UD-IQ4_XS has four shards, the pinned one three).
UNSLOTH_V31_SHARDS = {
    "Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf":
        (10946618, "b0d883bb9464021f31074145af74bfc2aa2d4bf0cf506f26f6f5107f4ca6ce17"),
    "Qwen3.8-Flash-Next-UD-IQ1_M-00002-of-00003.gguf":
        (39196009888, "bddcd79634ccddbb2e0951345c73c6ae8bd7b12594f55a2a529474701757391e"),
    "Qwen3.8-Flash-Next-UD-IQ1_M-00003-of-00003.gguf":
        (28800138432, "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6"),
    "Qwen3.8-Flash-Next-UD-Q2_0-00001-of-00003.gguf":
        (10946618, "38803c4e8d13077165aa9854502b3d4e7fef750e72a21d27a38ee42089f96ad6"),
    "Qwen3.8-Flash-Next-UD-Q2_0-00002-of-00003.gguf":
        (41738806688, "57af20207d5f18fa72fba503806d98d65f58ba3982fe55c89c6ed6e691471acc"),
    "Qwen3.8-Flash-Next-UD-Q2_0-00003-of-00003.gguf":
        (28800138432, "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6"),
    "Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf":
        (10946618, "91e8dc8f1428ce25d48fd1f711e4bf1f42215c5bac0f32eeb5608e6ec7ae91e9"),
    "Qwen3.8-Flash-Next-UD-IQ2_XS-00002-of-00003.gguf":
        (47021008288, "11a67c82f54a9edf683d6b708062de78d803921a18a6cb2bf1dd55949b4ef9bd"),
    "Qwen3.8-Flash-Next-UD-IQ2_XS-00003-of-00003.gguf":
        (28800138432, "05716382c280d999b81afff92b2d527fadbfd926ac225ffdc273f2d1d13331c6"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00004.gguf":
        (10946618, "297e174dc185f2bf3ab1856461059cb8a05aa741bc27edf42461cde87e5420af"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00002-of-00004.gguf":
        (49732486272, "a4782511810bb61bd6b3236607a0aad05b1bfa04b90356defd202bb17e7db16d"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00003-of-00004.gguf":
        (1037181344, "4dffc4a1cea7ad484f2d5c711b747840e459b040b636cc8ddaa524d01e60662e"),
    "Qwen3.8-Flash-Next-UD-Q2_K_XL-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00004.gguf":
        (10946618, "9d5644b371ffac921427c8021cc7242dab8e3ba37d7b48d1e79fe3770fbf9915"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00004.gguf":
        (49946027488, "aacaff3bdb9032c2c3be428167930809735f65d3cfb204eba07d85e421e7ddf4"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00004.gguf":
        (4869325888, "ef8f14e790ab471b363f146d5e8664eefbaf03e5776f317a1fb96a2bc7707b07"),
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00004.gguf":
        (10946618, "7666000ba91eb5527f8e12e0414de36766dc82a082876337f7827c2caa859143"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00004.gguf":
        (49742117248, "e954cb3044ae087a003ce8cc1631c9d40fe15ae64a8befcf4aa06457848c647c"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00004.gguf":
        (10911746208, "fcce26b1e5e6e54559a7e7c2f8d6e37067ec695ff1f14bce662810bde78abe1f"),
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf":
        (10946618, "93ff23a08e5cd7501f24fa16f5b415d5807d0175e04edd3ed6a7a41b601a4eef"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00004.gguf":
        (49604491008, "13d5dc8b93d039e4cfade5a9af7037f10c97f7ec4e53f75bd2e9ae34aae54aeb"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00004.gguf":
        (15256783680, "f6b9b5934908a2043732269dd740155d0f921d0763a1dcdd3452246311651f87"),
    "Qwen3.8-Flash-Next-UD-IQ4_XS-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
}
# UD-Q4_K_XL as re-sharded at UNSLOTH_V31_REVISION: the same 1224 tensors, the n-gram table in a shard of its own.  The file names are the pinned UD-Q4_K_XL's, so these are
# NOT downloaded: setup keeps fetching the pinned files, and takes this layout from files already here (--gguf-dir, --model-path, the app's "Use a local GGUF") by size and
# SHA-256 (fam_pin).  One set or the other: the shard boundaries differ, so the shards of the two cannot be mixed.
UNSLOTH_Q4_V31_SHARDS = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf":
        (10946618, "acc44040f74d0e55fb89c3967056cce5dbda1bcc52e909c3ba6fc72b5e9c0858"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf":
        (49720539552, "45792ae869072d8229384c51c7c73dd959e191505252b41dd71852d9746f5ce4"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf":
        (32803030176, "8c76e79d444b16a663d9c8db54e585ed3b1da13eebd2619c37b0c355da803c82"),
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf":
        (28800138432, "5de8b5dbf89409acc39468decafd4a4a95b13c29d88a14b0356107897829ec36"),
}
UNSLOTH_ENGINE = (0, 1, 32)     # the first engine setup configures for UD-Q4_K_XL (0.1.31 ran it by hand)
# Contexts past 262144 (the model's trained length) extend it by rope scaling: for the context it will
# serve the setup resolves the method (yarn, or one question when interactive) and derives the factor
# from the final context (final / 262144, at least 1) itself (below), keeps an explicit
# --rope-scaling/--rope-scale, and refuses an explicit --rope-scaling none there - the stock angles past
# the trained range are out of spec.
# 204800 (200K) sits between 128K and 256K: it is inside the trained 262144, so it needs no rope scaling and
# costs ~2.8 GB of 8-bit KV with IQ3_S (vs ~3.6 GB at 256K) - a middle step for PCs that cannot hold 256K.
CONTEXTS = [8192, 32768, 65536, 131072, 204800, 262144, 393216, 524288]
# The model families: the same architecture, weights in the same three GSQ-RCO sizes, different files.
FAMILIES = {
    "qwen": {"title": "Qwen3.8-Flash-Next", "by": "Qwen; GSQ-RCO quants by ISTA-DASLab",
             "about": "the original model",
             "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF") + "{q}/",
             "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "",
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
             "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next"},
    "swift": {"title": "Swift 1.5", "by": "UkisAI's fine-tune of Qwen3.8-Flash-Next",
              "about": "thinks much shorter (-63% thinking tokens, 1.8x sooner answers by its authors' numbers)",
              "hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "swift-",
              "mmproj_hf": hf("ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
              "mmproj": "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", "name": "swift-1.5",
              "license": "Swift Open License 1.0: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF"},
    # ISTA-DASLab's expert-pruned release: half of each layer's experts removed, chosen for code, agentic tool use and
    # vision; its shard 2 (the n-gram table) and vision encoder are the original's files, shared with it
    "coder": {"title": "Qwen3.8-Flash-Next Coder", "by": "ISTA-DASLab's coding version",
              "about": "half the experts (code, tools, images kept): needs ~32 GB of RAM, faster; weaker outside coding",
              "hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF") + "{q}/",
              "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf", "tag": "coder-",
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"),
              "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    # Unsloth's UD-IQ4_XS (three shards, #621), UD-Q4_K_XL (four), UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 (six) of the original
    # model (docs/QUANTS.md): regular choices since f-quants; the shard count and "vision" are per model (MODELS)
    "unsloth": {"title": "Qwen3.8-Flash-Next (Unsloth)", "by": "Unsloth's 1- to 8-bit quantizations",
                "about": "UD-IQ1_M, UD-Q2_0, UD-IQ2_XS, UD-Q2_K_XL, UD-IQ3_XXS, UD-Q3_K_XL, UD-IQ4_XS, UD-IQ4_XS-v31, "
                         "UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL or Q8_0 (68 to 188 GB downloads): every expert in "
                         "RAM from ~50 GB (UD-IQ1_M) to ~136 GB (Q8_0) of RAM, else part of them read from the SSD",
                "hf": hf("unsloth/Qwen3.8-Flash-Next-GGUF") + "{q}/",
                "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf", "shards": 4, "tag": "unsloth-",
                "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"),
                "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "name": "qwen3.8-flash-next-unsloth",
                "vision": True, "pack_args": ["--compat-bf16"],
                "sha256": {**UNSLOTH_V31_SHARDS, **UNSLOTH_SHARDS, **UNSLOTH_IQ4_XS_SHARDS, **UNSLOTH_BIG_SHARDS},
                "sha256_alt": {n: (pin,) for n, pin in UNSLOTH_Q4_V31_SHARDS.items()}},
    # MODEL-UPDATE Phase 4: the original model with NVFP4 experts, as ready-made packs (NVFP4_REPOS): the files come
    # as the engine reads them (no GGUF to pack), each size has its own repository ("repo" in MODELS)
    # agentionai's Gyro quantizations (see GYRO_FILES): after the regular families, never the default
    "gyro": {"title": "Qwen3.8-Flash-Next (Gyro)", "by": "Gyro quantizations by agentionai",
             "about": "Gyro-S (27.6 GB of memory) and Gyro-M (35 GB): the smallest sizes, for PCs with little memory; "
                      "58 and 92 GB downloads (the files include the n-gram table)",
             "hf": hf("agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"), "file": "Qwen3.8-Flash-Next-{q}.gguf", "shards": 1,
             "tag": "gyro-", "name": "qwen3.8-flash-next-gyro", "vision": True, "pack_args": ["--compat-bf16"],
             "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
             "sha256": GYRO_FILES,
             "license": "Qwen Community License 1.0 (Qwen3.8-Flash-Next's; the Gyro files are agentionai's quantization): "
                        "https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"},
    "nvfp4": {"title": "Qwen3.8-Flash-Next (NVFP4)", "by": "NVFP4 packs by sergqwer (strata-nvfp4)",
              "about": "4-bit NVFP4 experts, NVIDIA cards; a 128 GB download, ~80 GB of RAM to keep every expert in RAM",
              "tag": "nvfp4-", "name": "qwen3.8-flash-next-nvfp4", "ready": True,
              "mmproj_hf": hf("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"), "mmproj": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
              "mmproj_sha256": (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")},
}
MMPROJ = "mmproj-Qwen3.8-Flash-Next-BF16.gguf"
# Qwen3.6-35B-A3B (GGUF architecture qwen35moe), Unsloth's GGUFs with the MTP block: a family with an engine path of its
# own (src/models/qwen36; `strata` switches to it from the file's architecture).  No pack, no expert profile, no separate MTP
# draft layer: the engine reads the GGUF as it is, keeps every expert it can in VRAM and computes the rest on the CPU
# (the hybrid mode), and drafts with the file's own MTP block.  Not in FAMILIES (whose entries are Flash-Next files with
# packs): OWN_PATH_FAMILIES, offered in the same menu and installed by install_own_path.  NVIDIA (CUDA) and AMD (HIP)
# only: the Intel Arc (SYCL) engine refuses it.
QWEN36_FILES = {
    "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf": (22853663008, "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"),
    "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf": (32611711264, "35fce994cd36104a7dc1bd8a4bdf13778145664c00fdef6773aebc9246e5019c"),
    "Qwen3.6-35B-A3B-Q8_0.gguf": (37801097504, "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131"),
    "mmproj-F16.gguf": (899283584, "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160"),
}
OWN_PATH_FAMILIES = {
    "qwen36": {"title": "Qwen3.6-35B-A3B", "by": "Qwen; GGUFs with the MTP layer by Unsloth", "arch": "qwen35moe",
               "about": "35B mixture of experts with 3B active (256 experts, top-8): a smaller cousin of "
                        "Qwen3.8-Flash-Next with the same tokenizer",
               "hf": hf("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"), "file": "Qwen3.6-35B-A3B-{q}.gguf", "tag": "qwen36-",
               "mmproj_hf": hf("unsloth/Qwen3.6-35B-A3B-MTP-GGUF"), "mmproj": "mmproj-F16.gguf",
               "name": "qwen3.6-35b-a3b", "license": "Apache 2.0: https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
               # RAM: the engine keeps a copy of every expert in RAM when they do not all fit in VRAM (the hybrid mode),
               # so the file's size plus room for the OS and the rest (measured: UD-Q4_K_XL on a 46 GB PC with a 12 GB
               # card, 18.75 GiB of experts pinned)
               "models": {"UD-Q4_K_XL": {"about": "4-bit (Unsloth Dynamic), the smallest", "ram_gb": 32},
                          "UD-Q6_K_XL": {"about": "6-bit (Unsloth Dynamic), closer to the full model", "ram_gb": 44},
                          "Q8_0": {"about": "8-bit, the closest to the full model", "ram_gb": 52}},
               "sha256": QWEN36_FILES,
               # the engine's KV cache per context token: 10 attention layers + the MTP block, 2 KV heads x 256, K and
               # V in fp32 (strata-q36)
               "kv_bytes_per_token": 11 * 2 * 256 * 2 * 4,
               "mtp": 2,                     # drafts per step (greedy requests): measured best on the R9700
               # pictures: Unsloth's mmproj (the same projector family as Flash-Next's: qwen3vl_merger) through
               # strata-vision; a parallel slot's session: its KV cache (kv_bytes_per_token) and the recurrent state of
               # the 30 GDN layers (state_gb)
               "vision": True, "state_gb": 0.07}
}
PREVIEW_FAMILIES = OWN_PATH_FAMILIES             # the earlier name (tools and tests that import it)

# DeepSeek-V4.1-Flash (GGUF architecture deepseek41), Unsloth's UD-Q8_K_XL at the revision the port was checked against: a
# family with an engine of its own (strata-dsv41, src/models/deepseekv41, ported from Project Maya's maya-ds4, MIT) that is not a
# Flash-Next pack and not a single GGUF: nine shards (508.4 GB, the routed experts as MXFP4, the dense weights in BF16, and
# in shard 8 the 202.8 GB n-gram table that stays on the SSD), the engine's pack folder (tools/dsv41/pack_gguf.py: hash
# tables, tensor map, the tokenizer and the GGUF's own chat template, a few MB) beside them.  MULTI_FILE_FAMILIES, offered
# in the same menu and installed by install_dsv41.  Where it fits: strata_control/dsv41_fit.py (real numbers).  The DSpark
# drafter (dspark-...-Q8_0.gguf, 8.0 GB) and the vision tower's weights (mmproj-BF16.gguf, 1.0 GB) are pinned here too.
DSV41_REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
DSV41_SHARD = "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-{i:05d}-of-00009.gguf"
DSV41_FILES = {
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00001-of-00009.gguf": (5780589, "cc15d470afcb030558bbc4d8e2ccd840182e2e95de1a95198138efe63a5fd99f"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00002-of-00009.gguf": (50536121984, "9cb2047676bd20e448be2377b480080c97fb38fae117f4846e07de2c65bb8522"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00003-of-00009.gguf": (50536121984, "17d398dd327e7ff1991b382677814db75d970b074661757564b6f2ac3d0c8d95"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00004-of-00009.gguf": (50536122016, "7ea8da1143b9a6bf1424a361eb1b0d6f66c7ce79eac9711162f212cc4f39c319"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00005-of-00009.gguf": (50536122016, "1952da19a841176b46f9681c99ea57c674f8a8861e16d76fb35cec4befe039b2"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00006-of-00009.gguf": (50536122016, "d6711781d8effab79b410af9c40920bffcc0c28976e509bc1227ff0c6669c33a"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00007-of-00009.gguf": (36097230048, "2ebc8d198d03a14865e8a85df637285ceb5dd79b1ffc78424bd3c3e06df57823"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00008-of-00009.gguf": (202758032592, "12df293355efbd104b088a77e13ee22410fb1ec3f7ab3863947f09896e64bec7"),
    "DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL-00009-of-00009.gguf": (16835109952, "df33722d1dcf5ffe0347ed8406f7acb3b18061f4ded2c13c2b0a184b8d41c9ea"),
    "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf": (7976914240, "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"),
    "mmproj-BF16.gguf": (972208288, "9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9"),
}
# The OFFICIAL ISTA-DAS sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF at the revision above, a flat
# repository; Hub LFS sha256 = what the files hash to): IQ2_XXS is the 2.00 bpw set, IQ2_XS the 2.25 bpw set and IQ1_M
# the 1.75 bpw set.  Each is one GGUF split of 6 shards: shard 1 holds the metadata only (the Engram's hash parameters
# included, so pack_gguf needs no --engram-meta), shards 2-5 the tensors, and shard 6 is the ONE Engram table every set
# shares (110.6 GB, DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf: there is no file named ...-00006-of-00006.gguf; downloaded
# once into the shared folder).  The DSpark drafter comes from Unsloth's repository (pinned in DSV41_FILES above).
# This is the set a normal PC runs (64 GB RAM + a 12 GB card; docs/DEEPSEEK41.md); the UD-Q8_K_XL entry stays for the
# big machines.
DSV41_ISTA_REPO = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
DSV41_ISTA_ENGRAM = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
DSV41_ISTA_SHARD = "DeepSeek-V4.1-Flash-1017-GSQ-RCO-{quant}-{i:05d}-of-{n:05d}.gguf"
DSV41_DSPARK = "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"
DSV41_ISTA_FILES = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf": (5780704, "3e8ddd11441edd4bc19bacaef5dbe38818af2cbdf0820fb70f1d2d851fcc192a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00006.gguf": (39340012896, "be4ceb1caf1bed28e886860cf6a52adc8f5ca9eb0d2098ade39bb4179d48b323"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00006.gguf": (38698805888, "758c66ba65236e8ce0820c21a7b6f9b2947681df8cc509f672a77722e8191b9a"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00006.gguf": (39707055008, "04a5bea2b38eb576f49efc19c0a677768d6e2a8ea5772d72fb6cb79e1733a212"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00005-of-00006.gguf": (2969539840, "7c8dbee99890f05dbc420eba6f44393f5228d06248fe61e1ae8ed002449c7549"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00006.gguf": (5780704, "0182a2a1f42d319a0dbc6e34aa82a063880c069fe3aafbbea86018f386e64e5f"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00006.gguf": (39581533600, "5c88039fa640460a9fc3465a6dfcd162a4083dd738d55e5b52e025be14555a08"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00006.gguf": (39417972640, "0fddebfd05f0a36cfd107ba5dfd2547e64df61adbab39560a5a5f5521ca099a5"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00006.gguf": (39079187648, "c577a43c6e6e615838a1655a0b162541edeac6f36542f08c4a9593a70dcaf500"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00005-of-00006.gguf": (37141423776, "851d8605bf504962054d17ea8a0d9e2e0a228d085d396bd865a4de9f16d90fbd"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf": (5780704, "f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf": (39535781184, "25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf": (39736712000, "8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf": (39036908384, "9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf": (19923784832, "7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1"),
    "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf": (110595290816, "3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3"),
}
# The files of the sets' earlier layouts, which people downloaded before: the same weights, accepted from "files
# already here" (size and SHA-256 checked against these pins) and never downloaded.  Not in REPO_PINS: nothing is
# fetched under these names.
#  - 2026-10-24 to 10-28 (revision cb02422c): 4 shards a set (...-0000N-of-00004.gguf), tensors in shard 1 too;
#  - the first release (helcig/tmp-oct26-2, 2026-10-23): model-2.00bpw-0000N-of-00004.gguf, model-2.25bpw-...,
#    engram-IQ4_NL.gguf, usually in a 2.00bpw/ or 2.25bpw/ folder.  Its shards 2-4 are the same bytes as the 4-shard
#    layout's; shard 1 and the Engram differ in their header.
# Their shard 1 lacks the Engram's hash parameters: their packs read them from Unsloth's metadata shard
# (DSV41_ENGRAM_META).
DSV41_OLDER_FILES = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf": (39358830464, "6e3f0d5ff53d00428ec32c079db9e8255cb410ccaa435e5342aaa8a47fe1339d"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00002-of-00004.gguf": (39176113248, "34d68c99fdae9948f9f0af0dafe308e805ce0a78d2c48641b24cc57242fdc0af"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00003-of-00004.gguf": (39997334720, "a08468c83955e0c4d3a9eb3f7b86c45b7bbac55d99c4640039a296a8a4b327ea"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00004-of-00004.gguf": (2188841824, "b05f26a92325aa6845cebcb3ec1f93d2e65ebf89bdc3f805695960d2f0681832"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf": (39535776992, "4c65307992ab586c527ca1f6f64062aa96a33b23c02b7fed103f45f75e854798"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00002-of-00004.gguf": (39494617536, "933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00003-of-00004.gguf": (39423838496, "bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00004-of-00004.gguf": (36771591232, "e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf": (39075714624, "d46efb67ccffd8fb5d7f18b423896381b9e06d8491e161c370ddf3b05b07e050"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf": (39388449504, "cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf": (39246891616, "9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf": (20527837312, "9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059"),
    "model-2.00bpw-00001-of-00004.gguf": (39075713728, "78ab9efc46a26647c3e94b322c99cd7e2af261dd28b38f2712e128562db1ef47"),
    "model-2.00bpw-00002-of-00004.gguf": (39388449504, "cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70"),
    "model-2.00bpw-00003-of-00004.gguf": (39246891616, "9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3"),
    "model-2.00bpw-00004-of-00004.gguf": (20527837312, "9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059"),
    "model-2.25bpw-00001-of-00004.gguf": (39535776064, "ddd13aca16faa23b08f5196b5e0daddf90f04688a271f5889eb22fa7d7d65526"),
    "model-2.25bpw-00002-of-00004.gguf": (39494617536, "933461a974fcccee741bf2f9798201e72ab4aa3f8a952be56f849896ca6ea30e"),
    "model-2.25bpw-00003-of-00004.gguf": (39423838496, "bbf91fa346a85ebaf32dd4fd5c456b534d780d381e54f46ac90783a481373240"),
    "model-2.25bpw-00004-of-00004.gguf": (36771591232, "e5b2f8781fab688cd4bc3bc7a69651b5dbb90ed94965657c6bead8bb4774f3cf"),
    "engram-IQ4_NL.gguf": (110595290720, "1f17d8ae37e9df3bb45f7ef1b4fcd1706d7b759af0413c117bf8e92aed459281"),
}
# ISTA re-published shard 1 of each 4-shard set on 2026-10-26 with a larger chat template (images in tool results; their
# PATCHING.md, and what their patch_chat_template.py makes of a shard 1 already downloaded): the same weights, accepted
# from files already here by its own size and SHA-256, never fetched.
DSV41_NEW_SHARD1 = {
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00004.gguf": (39358841536, "a7be0d5a48e4d1c13c5535f9123d9131ab8964e5acf8a0c01d14970e4b7f4df9"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf": (39535788064, "d9b949d55e27242b8afc7c2928749b888372d4c39682f9a131670d1c2555155e"),
    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf": (39075725728, "2dfe9711eac1f7814aa59154d11e8f27739f8b96875dc1e667eac9f8c756e1fe"),
}
# A file's SHA-256 at an earlier revision of its repository, with the same name and size: accepted as that file.  The
# Engram of the 4-shard layout differs from today's in its header only (8 ranges of its tensor data compared equal,
# 2026-10-29).
DSV41_EARLIER_SHA256 = {DSV41_ISTA_ENGRAM: "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a"}
# a shared file -> its name in the first release
DSV41_FIRST_NAMES = {DSV41_ISTA_ENGRAM: "engram-IQ4_NL.gguf"}
DSV41_ENGRAM_META = DSV41_SHARD.format(i=1)       # only its metadata is read (pack_gguf --engram-meta)
DSV41_MMPROJ = "mmproj-BF16.gguf"     # the vision tower's weights (972 MB, pinned): packed into the model's pack when pictures are on (0.2.7)
# the files every official set shares (name -> where it comes from); they live in models/deepseek-v4.1-flash-shared, once
DSV41_SHARED = {
    DSV41_ISTA_ENGRAM: hf(DSV41_ISTA_REPO),
    DSV41_ENGRAM_META: hf(DSV41_REPO) + "UD-Q8_K_XL/",
    DSV41_DSPARK: hf(DSV41_REPO),
    DSV41_MMPROJ: hf(DSV41_REPO),
}
# Unsloth's own low-bit sets (2026-10-30, unsloth/DeepSeek-V4.1-Flash-1017-GGUF, UD-IQ1_S and UD-IQ1_M): shard 1 = metadata, shards 2-4 = the
# model, shard 5 = the Engram table (IQ4_NL, 110.6 GB; its data is byte-identical to ISTA's Engram file, so an ISTA Engram already here is used
# instead of fetching shard 5).  Same tensor names as the ISTA-DAS six-shard layout; dense weights in Q6_K / Q8_0 / Q5_K, router F32.  Opt-in
# model names, not the download default.
DSV41_UD_SHARD = "DeepSeek-V4.1-Flash-1017-{quant}-{i:05d}-of-00005.gguf"
DSV41_UD_FILES = {
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf": (5780868, "5197c309479c7be06e3612dc696d7f18e13e5942cfe52e70c3c586b7e585a5ac"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00002-of-00005.gguf": (49707332128, "fbfb546557a33f41332ffbed7f15fbed249b9f7796975d9db571c8cf3870f335"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00003-of-00005.gguf": (49292777280, "394b7a06e04d4b1b96b6028ec3d4dec49e4716900c26a493e8c2aa38a357ea1a"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00004-of-00005.gguf": (40803165856, "323a7f59a2938879eaac82b37c443c85151e518071ad44c79b1e4580c91d0ed7"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00005-of-00005.gguf": (110595290592, "771805301e695d72d268151dfc43043581ba904298574aa879d4349cbe67b020"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf": (5780868, "5197c309479c7be06e3612dc696d7f18e13e5942cfe52e70c3c586b7e585a5ac"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00002-of-00005.gguf": (49120285248, "e83af39db628614d252d49a316af8cd5911d78ab96a9e7a5997b392ece26eaf7"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00003-of-00005.gguf": (48899707008, "6caa1da6302b532374d55da6510fc92a9031a0ee1bd2bb5bf7defca2477f3d98"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00004-of-00005.gguf": (24358544704, "71c6c54dd1f20008e2130188cc655742d3738a365de4571d21a56de503b206b7"),
    "DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00005-of-00005.gguf": (110595290592, "771805301e695d72d268151dfc43043581ba904298574aa879d4349cbe67b020"),
}
DSV41_FILES.update(DSV41_UD_FILES)      # (the repository's pins: REPO_PINS)
# The ISTA-DAS Coder (2026-11-01; the official ISTA-DASLab repository since 2026-11-02): ISTA-DAS's coding variant of
# this model, "the layout of the IQ2_XXS set with half of the experts removed": 192 routed experts a layer instead of
# 384 (the router, the experts and the routing biases are narrower; every other tensor has the 2.00 set's name, shape
# and type).  Five model files in the folder IQ2_XXS/ (shard 1 = the metadata, shards 2-5 = the model; the Engram
# would be the sixth shard of the split) plus two files at the repository root: the Engram table (the SAME file as
# the unpruned release's Engram, same name, size and SHA-256: the shared Engram, read from the SSD per row and never
# held in memory, so a copy already here is used where it is) and a picture file (mmproj, whose tensors are the shared
# picture file's).  Its repository id, commit, file names, sizes and SHA-256 are in ONE place,
# data/dsv41/coder-set.json, written by tools/dsv41/pin_coder.py from the Hub: a new repository is
# `pin_coder.py NEW/REPO` and nothing else (strata_control/dsv41_fit.py and the app read the same file).
DSV41_CODER_JSON = Path(__file__).resolve().parent.parent / "data" / "dsv41" / "coder-set.json"
try:
    DSV41_CODER = json.loads(DSV41_CODER_JSON.read_text(encoding="utf-8"))
except (OSError, ValueError):
    DSV41_CODER = {}                    # a checkout or a package without the file: the set is simply not offered
DSV41_CODER_REPO = DSV41_CODER.get("repo", "")
DSV41_CODER_FILES = {n: (int(v[0]), str(v[1])) for n, v in (DSV41_CODER.get("files") or {}).items()}
DSV41_CODER_MODEL = DSV41_CODER.get("name", "ISTA-DAS-Coder")
if DSV41_CODER_REPO:
    HF_REVISIONS[DSV41_CODER_REPO] = DSV41_CODER["revision"]
# The Coder repository's own picture file has Unsloth's tensors under another header (another size and SHA-256): a
# copy of it already here is accepted as the shared picture file (find_shared / verify_sha256), and nothing is fetched
# under its name.  The picture file of the Coder's first (temporary) repository, helcig/tmp-oct26-4 @ 31b7f2dc, which
# the early Coder builds pinned (972,207,968 bytes, the same tensors under a third header), is accepted too: a tester
# of those builds has it in the shared folder, and a second 972 MB fetch for it is not wanted.
DSV41_EARLY_CODER_MMPROJ = (972207968, "a7cdd95589d0b58fdab59263b700657364520667e06f40b1f3828f0f206ba609")
DSV41_ALT_FILES = ({DSV41_MMPROJ: [DSV41_CODER_FILES[DSV41_CODER["mmproj"]], DSV41_EARLY_CODER_MMPROJ]}
                   if DSV41_CODER_REPO else {})
# ... and a copy of it under ITS OWN name (the official repository's mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf, beside
# the shards from `hf download` or a hard link) is found as the picture file too, so the plan and the fetch agree:
# nothing is downloaded when the person has it.
DSV41_ALT_NAMES = ({DSV41_MMPROJ: (DSV41_CODER["mmproj"],)}
                   if DSV41_CODER_REPO and DSV41_CODER.get("mmproj") not in (None, DSV41_MMPROJ) else {})
# The Coder repository's Engram file is the unpruned release's Engram under the same name (DSV41_ISTA_ENGRAM), so it
# never replaces that pin: one Engram file, on the SSD, shared by every set.
DSV41_ALL_FILES = {**DSV41_FILES, **DSV41_ISTA_FILES, **DSV41_OLDER_FILES,
                   **{n: v for n, v in DSV41_CODER_FILES.items() if n not in (DSV41_MMPROJ, DSV41_ISTA_ENGRAM)}}


def dsv41_also(name: str) -> tuple:
    """Gets the other SHA-256 values that are the same file as `name`: the one it had at an earlier revision of its
    repository (DSV41_EARLIER_SHA256) and those of its other published copies (DSV41_ALT_FILES)."""
    return tuple(x for x in (DSV41_EARLIER_SHA256.get(name), *(v[1] for v in DSV41_ALT_FILES.get(name, ()))) if x)


_UD_SHARDS = [DSV41_SHARD.format(i=i) for i in range(1, 10)]
DSV41_MAC_CONTEXT = 8192      # DeepSeek on a Mac keeps the 8K it was run at (nothing was run on a Mac at 128K)
DSV41_CONTEXT = 131072        # 128K like every model (owner, 2026-11-01; strata_control/default_context.py)
# DeepSeek's pictures on a Mac are decided by the installed engine (strata_control/dsv41_engine.mac_vision_state: the
# marker STRATA_DS4_METAL_VISION_V1), not by a switch here
MULTI_FILE_FAMILIES = {
    "deepseek41": {"title": "DeepSeek-V4.1-Flash", "by": "DeepSeek; the 2-bit GGUFs by ISTA-DAS, UD-Q8_K_XL by Unsloth", "arch": "deepseek41",
                   "about": "a big model that runs on a PC with a 12 GB card and 64 GB of RAM (the two official 2-bit versions: 257 or "
                            "274 GB of disk, about 5-7 tokens a second) or, in Unsloth's 508 GB version, on several graphics cards",
                   "hf": hf(DSV41_REPO) + "UD-Q8_K_XL/", "tag": "deepseek-v4.1-flash-", "name": "deepseek-v4.1-flash",
                   "license": "DeepSeek's model license: https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
                   "models": {"UD-Q8_K_XL": {"about": "Unsloth Dynamic 8-bit: the experts in MXFP4 as DeepSeek made them, the rest "
                                                       "in BF16 (the checkpoint's own precision)",
                              "ram_gb": 306, "shards": _UD_SHARDS}},
                   "sha256": DSV41_ALL_FILES, "vision": True}     # the engine's own tower (strata-dsv41 --vision on, the default), 0.2.7
}


def make_ista_set(quant: str, first_release: str | None = None) -> dict:
    """Builds the file part of an official ISTA-DAS set's entry in MULTI_FILE_FAMILIES (`quant`: the IQ type its shards
    are named after; `first_release`: the set's name in the first release, "2.00bpw", when it had one).  Its "layouts"
    are every shard list the set was published as, newest first: a folder holds the set when it holds every file of one
    of them (never a mix).  The first is what setup downloads ("shards"); the others' files are pinned in
    DSV41_OLDER_FILES."""
    layouts = [[DSV41_ISTA_SHARD.format(quant=quant, i=i, n=6) for i in range(1, 6)],
               [DSV41_ISTA_SHARD.format(quant=quant, i=i, n=4) for i in range(1, 5)]]
    if first_release:
        layouts.append(["model-%s-%05d-of-00004.gguf" % (first_release, i) for i in range(1, 5)])
    return {"recipe": "ista", "subdir": None, "legacy_subdir": first_release, "hf": hf(DSV41_ISTA_REPO),
            "shards": layouts[0], "layouts": layouts, "shared": [DSV41_ISTA_ENGRAM, DSV41_DSPARK]}


def make_unsloth_low_set(quant: str) -> dict:
    """Builds the file part of the entry of one of Unsloth's low-bit sets (DSV41_UD_FILES) in MULTI_FILE_FAMILIES: packed
    and checked like an official ISTA-DAS set ("recipe" udlow), from the set's shards 1-4; its shard 5 ("engram_shard") is
    the Engram, taken only when no copy of ISTA's Engram file is here."""
    shards = [DSV41_UD_SHARD.format(quant=quant, i=i) for i in range(1, 5)]
    return {"recipe": "udlow", "subdir": quant, "legacy_subdir": None, "hf": hf(DSV41_REPO) + quant + "/",
            "shards": shards, "layouts": [shards], "engram_shard": DSV41_UD_SHARD.format(quant=quant, i=5),
            "shared": [DSV41_ISTA_ENGRAM, DSV41_DSPARK]}



def make_coder_set() -> dict:
    """Builds the file part of the ISTA-DAS Coder's entry in MULTI_FILE_FAMILIES from data/dsv41/coder-set.json: packed
    like an official set ("recipe" coder) from its own shards, whose names are matched together with their pinned
    sizes ("generic_names"); "experts" is its experts per layer, for the fit."""
    shards = list(DSV41_CODER["shards"])
    subdir = DSV41_CODER.get("subdir") or None
    return {"recipe": "coder", "subdir": subdir, "legacy_subdir": None,
            "hf": hf(DSV41_CODER_REPO) + (subdir + "/" if subdir else ""), "shards": shards, "layouts": [shards],
            "engram_shard": DSV41_CODER["engram_shard"], "experts": DSV41_CODER["experts_per_layer"],
            "generic_names": True, "shared": [DSV41_ISTA_ENGRAM, DSV41_DSPARK]}

def pick_dsv41_layout(md: dict, has) -> list:
    """Picks the layout of a DeepSeek set's entry `md` (its "layouts") that a folder holds, `has(name)` saying whether a
    file is there: a complete one when there is one, else the one with the most files there, the newest on a tie (the
    download layout when the folder holds none).  Used by setup, the Terminal and the MCP server alike."""
    return min(md["layouts"], key=lambda names: (not all(map(has, names)), -sum(map(has, names))))


# The official sets first (a normal PC), Unsloth's 508 GB set last.  Each model: its shards ("shards", what setup
# downloads, and "layouts", every list of them a folder may hold), the sub-folder they are in at the repository (and in
# a --gguf-dir folder), the files shared with the other sets ("shared"; an older layout needs DSV41_ENGRAM_META too),
# the pack recipe and the shipped expert profile.
MULTI_FILE_FAMILIES["deepseek41"]["models"] = {
    "IST-DAS-2.00bpw": {
        "label": "ISTA-DAS 2.00 bpw",
        "about": "the smaller official version: 138 GB of model + 111 GB n-gram table + 8 GB drafter (257 GB of disk). Runs on a PC with "
                 "a 12 GB card and 40-64 GB of RAM at about 5-7 tokens a second; quality about 15.5% above the original model's "
                 "perplexity (code and agent work 5-9%)",
        "ram_gb": 40, "profile": "profile-2.00.bin", **make_ista_set("IQ2_XXS", "2.00bpw")},
    "IST-DAS-2.25bpw": {
        "label": "ISTA-DAS 2.25 bpw",
        "about": "the better official version: 155 GB of model + 111 GB n-gram table + 8 GB drafter (274 GB of disk, 17 GB more than "
                 "2.00). About the same speed; quality about 7.3% above the original model's perplexity (code and agent work 2-3%)",
        "ram_gb": 44, "profile": "profile-2.25.bin", **make_ista_set("IQ2_XS", "2.25bpw")},
    "IST-DAS-1.75bpw": {
        "label": "ISTA-DAS 1.75 bpw",
        "about": "the smallest official version: 121 GB of model + 111 GB n-gram table + 8 GB drafter (239 GB of disk, 18 GB less than 2.00). Runs "
                 "where 2.00 is tight (a 12 GB card and 32-40 GB of RAM). Smallest: fine for code and agent work, noticeably weaker on plain prose; pick 2.00 if it fits",
        "ram_gb": 34, "profile": "profile-2.00.bin", **make_ista_set("IQ1_M")},
    **({DSV41_CODER_MODEL: {
        "label": DSV41_CODER.get("label", "ISTA-DAS Coder"),
        "about": "ISTA-DAS's coding version of DeepSeek-V4.1-Flash: %d of the model's 384 experts in every layer are kept, the rest removed, with the 2.00 version's file layout and "
                 "tensor types: %.0f GB of model + 111 GB n-gram table + 8 GB drafter (about %.0f GB of disk). Run on a Mac (M5 Max) and a Ryzen AI Max at about 20 tokens a second. Against the original model (our test, 4,995 positions): "
                 "perplexity +48%%, KL 0.40 (2.00 bpw: +16%%, 0.17; 2.25 bpw: +8%%, 0.11)" % (DSV41_CODER["experts_per_layer"], sum(DSV41_CODER_FILES[n][0] for n in DSV41_CODER["shards"]) / 1e9,
                                                      (sum(DSV41_CODER_FILES[n][0] for n in DSV41_CODER["shards"]) + 110_595_290_816 + 7_976_914_240) / 1e9 + 6),
        "ram_gb": 24, "profile": DSV41_CODER.get("profile", "profile-coder.bin"), **make_coder_set()}} if DSV41_CODER_REPO else {}),
    "UD-IQ1_S": {
        "label": "Unsloth UD-IQ1_S",
        "about": "Unsloth's smallest dynamic version: 122 GB of model (experts in IQ1_S / IQ2_XXS, attention in Q6_K) + the 111 GB n-gram table + 8 GB drafter. Same size class as ISTA-DAS 1.75 bpw. A 1-bit version: expect weaker answers than the 2.00 and 2.25 versions on long, detailed or careful work (short answers, code and tool calls looked fine in our check; its quality is not measured against the original yet). Pick 2.00 or 2.25 if they fit",
        "ram_gb": 36, "profile": "profile-2.00.bin", **make_unsloth_low_set("UD-IQ1_S")},
    "UD-IQ1_M": {
        "label": "Unsloth UD-IQ1_M",
        "about": "Unsloth's next dynamic version: 140 GB of model (experts in IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S, attention in Q6_K) + the 111 GB n-gram table + 8 GB drafter. Same size class as ISTA-DAS 2.00 bpw. Part of its experts are 1-bit: expect somewhat weaker answers than the 2.25 version on long, detailed or careful work (short answers, code and tool calls looked fine in our check; its quality is not measured against the original yet)",
        "ram_gb": 40, "profile": "profile-2.00.bin", **make_unsloth_low_set("UD-IQ1_M")},
    "UD-Q8_K_XL": {
        "label": "Unsloth UD-Q8_K_XL",
        "about": "Unsloth Dynamic 8-bit: the experts in MXFP4 as DeepSeek made them, the rest in BF16 (the checkpoint's own precision); "
                 "508 GB, for machines with hundreds of GB of RAM or several cards",
        "ram_gb": 306, "recipe": "ud", "subdir": "UD-Q8_K_XL", "shards": _UD_SHARDS, "layouts": [_UD_SHARDS], "shared": []},
}

# Every model file Strata downloads has a fixed size and SHA-256: (repository, file name) -> (bytes, sha256).  download()
# checks the file against it as soon as it is whole (a wrong file is deleted and setup stops), and stops before the
# download when the server's size is not the pinned one.  The files are fetched from the commits in HF_REVISIONS.
REPO_PINS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": ISTA_FILES,
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": SWIFT_FILES,
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": CODER_FILES,
    UNSLOTH_REPO: {**UNSLOTH_V31_SHARDS, **UNSLOTH_SHARDS, **UNSLOTH_IQ4_XS_SHARDS, **UNSLOTH_BIG_SHARDS},
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": QWEN36_FILES,
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": GYRO_FILES,
    "unsloth/DeepSeek-V4.1-Flash-1017-GGUF": DSV41_FILES,
    "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF": DSV41_ISTA_FILES,
}
# the files of a repository at a commit other than HF_REVISIONS' (UNSLOTH_V31_REVISION): (repository, commit) -> name ->
# (bytes, sha256); setup's file_pin looks here first
REV_PINS = {(UNSLOTH_REPO, UNSLOTH_V31_REVISION): {**UNSLOTH_V31_SHARDS, **UNSLOTH_Q4_V31_SHARDS}}
