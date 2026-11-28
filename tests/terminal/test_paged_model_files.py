"""Checks model-file arguments without loading weights or starting an engine."""
from strata_common.paged_launch import translate_args
from strata_common.user_args import find_model_file
from strata_common import user_args, tune_knobs
from strata_control import arg_layers


def test_ready_pack_preserves_separate_dense_embedding_and_ple_files():
    args = ["--pack", "nvfp4/pack", "--native", "dense.gguf",
            "--native-dense-gguf", "dense.gguf", "--native-dense-gguf", "other.gguf",
            "--embd-gguf", "embedding.gguf", "--ple-gguf", "ple.gguf"]
    assert translate_args({"exe": "strata-q38.exe"}, args) == args


def test_canonical_pack_preserves_source_model_for_native_projections():
    args = ["--pack", "packs/q2_0", "--native", "q2_0-00001-of-00002.gguf",
            "--ple-gguf", "q2_0-00002-of-00002.gguf"]
    assert translate_args({"exe": "strata-q38.exe"}, args) == args


def test_model_file_is_read_under_either_name_and_reaches_qwen_under_its_old_one():
    assert translate_args({"exe": "strata-q38.exe"}, ["--model", "m.gguf"]) == ["--native", "m.gguf"]
    assert find_model_file(["--pack", "p", "--native", "a.gguf", "--model", "b.gguf"]) == "b.gguf"
    assert find_model_file(["--model", "b.gguf", "--native", "a.gguf"]) == "a.gguf"
    assert find_model_file(["--pack", "p"]) is None


def test_prompt_chunk_reaches_qwen_under_its_old_name_and_deepseek_unchanged():
    assert translate_args({"exe": "strata-q38.exe"}, ["--prefill-chunk", "4096"]) == ["--prefill", "4096"]
    assert translate_args({"exe": "strata-ds41.exe", "arch": "deepseek41"}, ["--prefill-chunk", "4096"]) == ["--prefill-chunk", "4096"]


def test_prompt_order_reaches_both_paged_engines():
    args = ["--prompt-order", "layer"]
    for exe in ("strata-q38.exe", "strata-ds41.exe"):
        assert translate_args({"exe": exe}, args) == args


def test_model_file_edits_replace_or_remove_either_spelling():
    for old, new in (("--native", "--model"), ("--model", "--native")):
        args = [old, "old.gguf", "--native-dense-gguf", "dense.gguf"]
        expected = ["--native-dense-gguf", "dense.gguf", "--native", "new.gguf"]
        assert user_args.merge(args, [new, "new.gguf"]) == expected
        assert user_args.set_flag(args, new, "new.gguf") == expected
        assert tune_knobs.apply_to_argv(args, {new: "new.gguf"}) == expected
        assert user_args.set_flag(args, new, None) == expected[:2]
        assert tune_knobs.apply_to_argv(args, {new: None}) == expected[:2]
        cfg = {"args": args, "user_args": [old, "old.gguf"]}
        arg_layers.layer(cfg, {new: "new.gguf"})
        assert cfg["args"] == expected
        assert cfg["user_args"] == ["--native", "new.gguf"]
        arg_layers.layer(cfg, {new: None})
        assert cfg["args"] == expected[:2]


def test_mixed_model_flags_keep_only_the_last_file_during_edits_and_translation():
    args = ["--model", "old.gguf", "--native", "new.gguf"]
    assert user_args.merge(args, []) == ["--native", "new.gguf"]
    assert translate_args({"exe": "strata-q38.exe"}, args) == ["--native", "new.gguf"]
    assert user_args.capture({"args": ["--model", "new.gguf"],
                              "args_written": ["--native", "old.gguf"],
                              "user_args": ["--native", "old.gguf"]}) == ["--native", "new.gguf"]
    assert find_model_file(["--native-dense-gguf", "dense.gguf", "--native-something", "x"]) is None
