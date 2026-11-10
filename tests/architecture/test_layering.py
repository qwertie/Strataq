"""Checks that code in the new layout (docs/ARCHITECTURE.md) uses only the layers below it.

    python -m pytest tests/architecture

C++ is checked by its `#include "strata/..."` lines (not by includes relative to the file), Python by its absolute
imports and by what it loads at run time (not in tests, which may set up any part of the system). Includes of the old
folders that the refactor is still emptying (OLD_CPP_FOLDERS) are allowed until they are gone.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CPP_SUFFIXES = {".cpp", ".cc", ".c", ".hpp", ".h", ".inc", ".inl", ".cu", ".cuh"}
CPP_INCLUDE = re.compile(r'^\s*#\s*include\s*"strata/([^"]+)"', re.MULTILINE)
# Folders under src/ and include/strata/ that the refactor empties and deletes (docs/ARCHITECTURE.md#old-folders)
OLD_CPP_FOLDERS = {"core"}
# Headers still in an old folder whose new place is decided (docs/UPSTREAM_PORTING.md#path-map): an include of one is
# checked as an include of its new place
OLD_CPP_HEADERS_MOVING_TO = {"core/geometry.hpp": "models/common/geometry.hpp"}
# C++ files (relative to src/ or include/strata/) that still include a higher layer, and why. Remove an entry once its
# file no longer does.
KNOWN_UPWARD_INCLUDES = {
    "common/artifact/gguf_reader.hpp": "reads a model file's shape into core/geometry.hpp's ModelGeometry",
    "common/kernels/cpu/expert.hpp": "sizes the CPU experts by the compiled geometry (STRATA_GEOMETRY)",
    "common/kernels/cpu/native_expert.hpp": "sizes the CPU pool's activations by the compiled geometry (STRATA_GEOMETRY)",
    "common/kernels/prefill/moe_fused.hpp": "sizes the fused prompt experts by the compiled geometry (STRATA_GEOMETRY)",
}
# The new C++ folders (relative to src/ or include/strata/) by layer. Files in one folder, including its subfolders,
# may include each other; folders in one layer may not, except as CPP_SAME_LAYER_USES allows. A new model family
# is added here.
CPP_LAYERS = {"base": 0, "base/gpu": 0, "base/platform": 0, "common": 1, "models/common": 2,
              "models/deepseekv41": 3, "models/qwen36": 3, "models/qwen38": 3, "program": 4}
CPP_SAME_LAYER_USES = {"base/gpu": {"base/platform"}}
# Python packages and scripts by layer; packages in one layer must not import each other
PY_LAYERS = {"strata_base": 0, "strata_common": 1, "strata_server": 2, "strata_control": 2, "strata_cli": 3,
             "setup": 4, "tools": 4}
# The packages whose imports are checked
CHECKED_PY_PACKAGES = ("strata_base", "strata_common", "strata_control", "strata_server", "strata_cli")
# Calls that load a module or run a file at run time; a scope (function or module) that makes one and names setup.py or
# a tools/ module in a string loads that layer
PY_RUNTIME_LOADERS = {"import_module", "__import__", "spec_from_file_location", "find_spec", "run_path", "run_module"}
# The scripts in tools/ by the name an import uses (some callers put tools/ on sys.path)
TOOL_MODULES = {path.stem for path in (ROOT / "tools").glob("*.py")}
# Files of the checked packages that still import a package of their own or a higher layer, or load setup.py or tools/
# at run time, and why. Remove an entry once its file no longer does.
KNOWN_UPWARD_LOADS = {
    "strata_cli/launch/catalog.py": "runs tools/strata_mcp.py's Strata object, the app's model list and fit code",
    "strata_control/calibration.py": "calls setup.py's hardware_key, runs tools/calibrate.py and strata_tokenizer.py",
    "strata_control/drafting_fetch.py": "runs tools/dflash2_pack.py to pack the downloaded drafter",
    "strata_control/engine_update.py": "upstream's engine update: reads the Terminal's launch folder, access and settings (strata_cli/launch)",
    "strata_control/intel_app.py": "upstream's Intel app install: adopts the model with strata_cli/launch/adopt.py and its card list",
    "strata_control/setup_config.py": "setup.py's run-config lines (moved by upstream): applies tools/calibrate.py's tuned settings",
    "strata_common/engine_component.py": "upstream's engines of model shapes: strata_control/shape_engine.py (it needs "
                                          "the feed and the updater); the saved backend choice: discovery.py's settings file",
    "strata_server/server.py": "imports tools/strata_tokenizer.py, the tokenizer, and strata_control/runtime.py for the "
                               "runtime folder's activity lease (upstream's #4463)",
}


def test_cpp_includes_follow_layers():
    problems = [f"{path.relative_to(ROOT).as_posix()}: {problem}" for path, own, problem in find_cpp_problems()
                if own not in KNOWN_UPWARD_INCLUDES]
    assert not problems, "\n".join(problems)


def test_known_upward_includes_are_still_there():
    upward = {own for path, own, problem in find_cpp_problems()}
    assert not set(KNOWN_UPWARD_INCLUDES) - upward, "remove these from KNOWN_UPWARD_INCLUDES"


def test_python_imports_follow_layers():
    problems = []
    for package, path in find_checked_py_files():
        name = path.relative_to(ROOT).as_posix()
        for target in find_imported_packages(path.read_bytes(), name):
            if (problem := find_python_problem(package, target)) and name not in KNOWN_UPWARD_LOADS:
                problems.append(f"{name}: {problem}")
    assert not problems, "\n".join(problems)


def test_known_upward_loads_are_still_there():
    upward = {path.relative_to(ROOT).as_posix() for package, path in find_checked_py_files()
              if any(find_python_problem(package, target)
                     for target in find_imported_packages(path.read_bytes(), str(path)))}
    assert not set(KNOWN_UPWARD_LOADS) - upward, "remove these from KNOWN_UPWARD_LOADS"


def test_rules_catch_violations():
    assert find_cpp_problem("models/qwen38/session.cpp", "models/common/geometry.hpp") is None
    assert find_cpp_problem("models/qwen38/session.cpp", "core/layer.hpp") is None
    assert find_cpp_problem("program/generate.cpp", "models/qwen36/engine.hpp") is None
    assert find_cpp_problem("common/state/pool.cpp", "models/qwen38/session.hpp")
    assert find_cpp_problem("base/gpu/device.cu", "common/state/pool.hpp")
    assert find_cpp_problem("models/qwen36/engine.cu", "models/qwen38/session.hpp")
    assert find_cpp_problem("base/gpu/device.cu", "base/platform/numa.hpp") is None
    assert find_cpp_problem("base/platform/numa.cpp", "base/gpu/device.hpp")
    assert find_cpp_problem("common/state/pool.cpp", "common/spec/draft.hpp") is None
    assert find_cpp_problem("common/state/pool.cpp", "nonexistent/x.hpp")
    assert find_cpp_problem("models/newfamily/engine.cu", "common/state/pool.hpp")
    assert find_cpp_problem("models/qwen38/session.cpp", "core/geometry.hpp") is None
    assert find_cpp_problem("common/kernels/cpu/expert.hpp", "core/geometry.hpp")
    assert find_python_problem("strata_common", "strata_base") is None
    assert find_python_problem("strata_server", "strata_common") is None
    assert find_python_problem("strata_base", "strata_common")
    assert find_python_problem("strata_server", "strata_control")
    assert find_python_problem("strata_control", "strata_server")
    assert find_python_problem("strata_control", "tools")
    assert find_python_problem("strata_control", "setup")
    loads = {
        'import importlib\ndef f():\n    return importlib.import_module("setup")': {"importlib", "setup"},
        'import importlib.util\nspec = importlib.util.spec_from_file_location("x", ROOT / "setup.py")':
            {"importlib", "setup"},
        'def f():\n    return __import__("tools.calibrate")': {"tools"},
        'import runpy\nrunpy.run_path(str(ROOT / "tools" / "x.py"))': {"runpy", "tools"},
        'import strata_tokenizer': {"tools"},
        'import importlib\nimportlib.import_module("json")': {"importlib"},
        'def f():\n    return "setup.py"\ndef g(importlib):\n    return importlib.import_module(name)': set(),
    }
    for code, expected in loads.items():
        assert set(find_imported_packages(code, "<rule test>")) == expected, code


def find_cpp_files():
    """Yields each C++ file in the new folders with its path relative to src/ or include/strata/, which is how
    an include names it."""
    for base in (ROOT / "src", ROOT / "include" / "strata"):
        for folder in sorted({folder.split("/")[0] for folder in CPP_LAYERS}):
            for path in sorted((base / folder).rglob("*")):
                if path.suffix in CPP_SUFFIXES:
                    yield path, path.relative_to(base).as_posix()


def find_cpp_problems():
    """Yields (path, its path relative to src/ or include/strata/, what is wrong) for each include of a C++ file in the
    new folders that the layers do not allow."""
    for path, own in find_cpp_files():
        for target in CPP_INCLUDE.findall(path.read_text(encoding="utf-8", errors="replace")):
            if problem := find_cpp_problem(own, target):
                yield path, own, problem


def find_cpp_problem(own: str, target: str) -> str | None:
    """Gets what is wrong with file `own` including `target` (both relative to src/ or include/strata/), or
    None if the layers allow it."""
    own_folder = get_cpp_folder(own)
    target_folder = get_cpp_folder(OLD_CPP_HEADERS_MOVING_TO.get(target, target))
    if own_folder is None:
        return "is in no folder of CPP_LAYERS"
    if target_folder is None:
        if target.split("/", 1)[0] in OLD_CPP_FOLDERS:
            return None
        return f"includes strata/{target}, which is in no folder of CPP_LAYERS"
    if CPP_LAYERS[target_folder] > CPP_LAYERS[own_folder]:
        return f"includes strata/{target} from a higher layer"
    if (CPP_LAYERS[target_folder] == CPP_LAYERS[own_folder] and target_folder != own_folder
            and target_folder not in CPP_SAME_LAYER_USES.get(own_folder, ())):
        return f"includes strata/{target}, which {own_folder}/ may not use"
    return None


def get_cpp_folder(path: str) -> str | None:
    """Gets the longest CPP_LAYERS key that contains `path` (relative to src/ or include/strata/), or None
    for an old folder."""
    return max((folder for folder in CPP_LAYERS if path.startswith(folder + "/")), key=len, default=None)


def find_checked_py_files():
    """Yields (package, path) for each Python file of CHECKED_PY_PACKAGES that is not a test."""
    for package in CHECKED_PY_PACKAGES:
        for path in sorted(p for p in (ROOT / package).rglob("*.py") if not p.name.startswith("test_")):
            yield package, path


def find_imported_packages(source: bytes | str, name: str):
    """Yields the top-level package of each import in Python source, and "setup" or "tools" for each scope that loads
    setup.py or a tools/ module at run time (PY_RUNTIME_LOADERS). A tools/ script imported by its own name counts as
    "tools"."""
    tree = ast.parse(source, name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (get_import_layer_name(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield get_import_layer_name(node.module)
    for scope in [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]:
        nodes = list(iter_scope_nodes(scope))
        if any(isinstance(n, ast.Call) and get_called_name(n.func) in PY_RUNTIME_LOADERS for n in nodes):
            yield from {layer for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)
                        if (layer := get_named_upper_layer(n.value))}


def get_import_layer_name(module: str) -> str:
    """Gets the PY_LAYERS name an import of `module` belongs to: its top-level package, or "tools" for a tools/
    script."""
    top = module.split(".")[0]
    return "tools" if top in TOOL_MODULES else top


def iter_scope_nodes(scope: ast.AST):
    """Yields the nodes of a module or function, leaving out the functions defined inside it (they are scopes too)."""
    for child in ast.iter_child_nodes(scope):
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield child
            yield from iter_scope_nodes(child)


def get_called_name(func: ast.expr) -> str | None:
    """Gets the last name of a called expression: `import_module` for `importlib.import_module(...)`."""
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def get_named_upper_layer(text: str) -> str | None:
    """Gets "setup" or "tools" when a string names setup.py or something in tools/ (a module, a script, the folder)."""
    if text in ("setup", "setup.py"):
        return "setup"
    if text == "tools" or text.startswith(("tools.", "tools/")) or (text.endswith(".py") and text[:-3] in TOOL_MODULES):
        return "tools"
    return None


def find_python_problem(package: str, target: str) -> str | None:
    """Gets what is wrong with `package` importing `target`, or None if the layers allow it."""
    if target == package or target not in PY_LAYERS:
        return None
    if PY_LAYERS[target] >= PY_LAYERS[package]:
        return f"imports {target}, which is not in a lower layer"
    return None
