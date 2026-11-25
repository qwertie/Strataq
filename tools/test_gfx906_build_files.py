"""#3172 #3271: the gfx906 (HIP wave64 compat) build stays buildable. The real check is a compile for gfx906 (done on a box with ROCm: the
engine targets link); these cover what a text check can: the pieces of CMakeLists.txt / cmake/*.cmake a 0.1.44+ gfx906 build was missing."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CMakeLists.txt is cut into cmake/*.cmake macros (cleanup 5.4): the text checks read all of them
HIP_TEXT = (ROOT / "cmake" / "backends" / "hip.cmake").read_text(encoding="utf-8").replace("\r\n", "\n")
TEXT = "\n".join(p.read_text(encoding="utf-8").replace("\r\n", "\n")
                 for p in [ROOT / "CMakeLists.txt", *sorted((ROOT / "cmake").glob("*.cmake")), *sorted((ROOT / "cmake" / "backends").glob("*.cmake"))])


class Gfx906CMake(unittest.TestCase):
    def test_qwen35_is_not_built_for_gfx906(self):
        # q36_engine.cu calls cublasSgemm / cublasGemmStridedBatchedEx, which the wave64 compat layer does not map: a gfx906 build leaves
        # the library and strata-q36 out, so the rest links
        block = re.search(r"if\(NOT STRATA_HIP_GFX906\)\n  add_library\(strata_qwen35.*?"
                          r"target_link_libraries\(strata-q36 PRIVATE strata_qwen35[^\n]*\n  endif\(\)", TEXT, re.S)
        self.assertIsNotNone(block)
        self.assertIn("add_executable(strata-q36 ", block.group(0))

    def test_the_w4a4_target_gets_the_hip_definition(self):
        self.assertRegex(TEXT, r"if\(TARGET strata_mmq_w4a4\)\n\s+target_compile_definitions\(strata_mmq_w4a4 PRIVATE GGML_USE_HIP\)")

    def test_overlay_sources_and_the_f16_stub_are_in_the_gfx906_build(self):
        tail = HIP_TEXT[HIP_TEXT.index("# HIP gfx906: every CUDA source"):]
        self.assertIn("${_strata_tq_overlay_dir}/quantize.cu", tail)
        self.assertIn("mmq-instance-${_t}.cu", tail)
        self.assertIn("target_sources(strata_mmq PRIVATE "
                      "${CMAKE_CURRENT_SOURCE_DIR}/src/common/kernels/prefill/moe_f16_kq.cu)", tail)
        self.assertLess(tail.index("moe_f16_kq.cu"), tail.index("set_source_files_properties(${_strata_cu} PROPERTIES LANGUAGE HIP)"))


if __name__ == "__main__":
    unittest.main()
