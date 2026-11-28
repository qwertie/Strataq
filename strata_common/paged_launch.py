"""Adapts saved server settings for the page-pool engines (strata-q38, strata-ds41), and maps engine programs that
are no longer shipped to the ones that replaced them."""
from pathlib import Path

from . import draft_layer, user_args

#: The run config "arch" of Qwen3.6-35B-A3B, the one family whose engine (strata-q36) is not a page-pool engine.
QWEN36_ARCH = "qwen35moe"


def is_paged_engine(cfg: dict) -> bool:
    """Tells whether a run config starts strata-q38 or strata-ds41: every family but Qwen3.6 (by its "arch" or by
    strata-q36 named as the program)."""
    return cfg.get("arch") != QWEN36_ARCH and get_engine_name(cfg) != "strata-q36"


def is_deepseek(cfg: dict) -> bool:
    """Tells whether a run config is DeepSeek V4.1's, by its "arch" or by the program it names."""
    return cfg.get("arch") == "deepseek41" or get_engine_name(cfg) in ("strata-ds41", "strata-dsv41")


def get_current_engine_path(exe: str, cfg: dict) -> str:
    """Gets the program to start in place of `exe`, an engine path resolved from run config `cfg`: a saved config may
    still name a program that is no longer shipped (`strata`, `strata-dsv41`); the program that replaced it sits in
    the same engine folder."""
    path = Path(exe)
    name = path.stem.lower()
    if name == "strata":
        new = "strata-q36" if cfg.get("arch") == QWEN36_ARCH else "strata-q38"
    elif name == "strata-dsv41":
        new = "strata-ds41"
    else:
        return exe
    return str(path.with_name(new + path.suffix))


def translate_args(cfg: dict, args: list[str]) -> list[str]:
    """Keeps supported options and reports settings with no page-pool equivalent."""
    deepseek = is_deepseek(cfg)
    if not deepseek:
        args = user_args.flat(user_args.groups(args))
    model_off = cfg.get("mtp") is False or cfg.get("drafting") == "off"
    if model_off:
        args, _ = draft_layer.strip(args)
    lookup_values = {"--suffix-draft": "0", "--lookup-chain": "0"}
    # Both paged families use the shared stage scheduler, including live batch-group sizing.
    common = set("""pack max-context batch batch-groups vram-reserve-mib reserve-mib eos-ids eos
        expert-cache expert-profile layer-split split-device
        adapt-every adapt-swaps adapt-min-gain adapt-decay expert-popularity-decay expert-half-life
        route-tail-skip route-prior-pct draft-confidence-pct draft-cumulative-pct
        tokens-min tokens-max conversation-tokens vram-conversations gpu-mib ram-mib ssd-pause-mib
        rows lookahead concurrent pause-dir copy-back conversation-idle-seconds decode-share-percent prompt-turn-ms
        drafter draft-rule remote-experts save-on-shutdown prompt-order step-profile
        suffix-draft mtp-max-t
        lookup-chain lookup-chain-min""".split())
    model = set(("""vision gpus cpu-threads dspark dspark-conf experts dspark-block prefill-chunk threads"""
                 if deepseek else """native native-dense-gguf embd-gguf ple-gguf kv prefill mtp spec-min-p
                 gpu pcie-frac pool-cores aux-cpus pool-affinity vram-reserve-later-mib""").split())
    switches = {"serve"} | ({"no-expert-cache"} if deepseek else
                 {"vision", "no-ple", "no-prefill-borrow", "no-host-worker", "no-ple-prefetch", "stop-eos"})
    # Installed engines may predate shared option names, so emit each family's original spelling.
    aliases = {"--slots": "--batch"}
    aliases.update({"--gpu": "--gpus", "--vram-reserve-mib": "--reserve-mib", "--eos-ids": "--eos"} if deepseek else
                   {"--gpus": "--gpu", "--resident-budget-gib": "--ram-mib", "--model": "--native",
                    "--reserve-mib": "--vram-reserve-mib", "--eos": "--eos-ids", "--prefill-chunk": "--prefill"})
    supported = common | model | switches
    out, omitted = [], []
    i = 0
    while i < len(args):
        flag = str(args[i])
        end = i + 1
        while end < len(args) and not str(args[end]).startswith("--"):
            end += 1
        values = args[i + 1:end]
        target = aliases.get(flag, flag)
        name = target.removeprefix("--")
        if flag == "--resident-budget-gib" and "--ram-mib" in args:
            omitted.extend([flag, *values])
        elif name in supported:
            if name in switches:
                out.append(target)
            elif values:
                if flag == "--resident-budget-gib":
                    values = [str(int(float(values[0]) * 1024))]
                out.extend([target, *values])
                if target in lookup_values:
                    lookup_values[target] = values[0]
            elif name == "vision" and deepseek:
                out.extend([target, "on"])
            else:
                raise ValueError(f"{flag} needs a value")
        else:
            omitted.extend([flag, *values])
        i = end
    if model_off:
        out += ["--drafter", "off"]
        if cfg.get("drafting") == "off" or not any(int(value) > 0 for value in lookup_values.values()):
            out += ["--lookahead", "0"]
        if deepseek:
            out += ["--dspark", "none"]
    if cfg.get("batchMtp"):
        print("[strata] paged engine: leaving out unsupported config settings: batchMtp", flush=True)
    if omitted:
        print("[strata] paged engine: leaving out unsupported settings: " + " ".join(omitted), flush=True)
    return out


def get_engine_name(cfg: dict) -> str:
    """Gets the lower-case executable basename from config paths saved on either operating system."""
    return Path(str(cfg.get("exe", "")).replace("\\", "/")).stem.lower()
