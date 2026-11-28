# Credits and licenses

Strataq is a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and its contributors. Both are under the
[MIT License](LICENSE). The model files are not part of this repository; each model's own license applies to its files.

## Models

- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. Quantizations:
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
  and the Coder,
  [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF)
  (Apache-2.0 per its card; its support in Strata came from @pjgmobile's PR #54);
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) by UkisAI; the
  [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) files (their support follows
  [eddoursul/Strata](https://github.com/eddoursul/Strata)).
- Qwen3.6-35B-A3B by the Qwen team, in
  [Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF).
- DeepSeek-V4.1-Flash by DeepSeek: the 2-bit GGUFs by ISTA-DASLab and
  [UD-Q8_K_XL](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) by Unsloth, under DeepSeek's model license.

## Code and data

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the i-quant formats, the GPU dot products and
  dequantizers transcribed in `src/common/kernels/cuda/iq_kernels.cu`, the CPU backend linked for the i-quant experts,
  the `mtmd` library behind the image encoder (`tools/vision/`), and `gguf-py` used by the tools. See
  `third_party/ggml/LICENSE` and `third_party/strata-ggml/LICENSE`.
- [Project Maya](https://github.com/mw00/project-maya) by Peasant Smith (mw00) and its contributors (MIT): the
  DeepSeek-V4.1 engine. See `third_party/project-maya/NOTICE.md`.
- [libstrix](https://github.com/asf0/libstrix) by asf0 (MIT): the workgroup barrier of the gfx1151 fused prompt experts.
  See `third_party/libstrix/NOTICE.md`.
- Ideas from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen).
- The web page's font: [Outfit](https://github.com/Outfitio/Outfit-Fonts) (SIL Open Font License 1.1, see
  `strata_server/web/fonts/OFL.txt`). Math is drawn by [KaTeX](https://katex.org) (MIT, vendored in
  `strata_server/web/vendor/katex`). The Monitor tab started from @code-martin's dashboard idea (PR #22). The chat
  page's wide-window layout, svg/html preview, math, Load/Unload, saved chats, edit and rewind came from PRs #3176
  (@mnott), #1628 (@thoives), #2972 and #2974 (@CC-David-CC), #3550 (@sakuumomo), #1787 (@Scotty-Hudson) and #2615
  (@uppet), and the rewind choice and sending while answering from issues #3823 and #4004 (@xvcezrqpbxtpzblrt).
- The experimental speed projection's vector (`data/experimental-speed-projection/`): Qwen Community License 1.0, made
  from the model's activations (see its README).
