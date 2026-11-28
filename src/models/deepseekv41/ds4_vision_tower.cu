// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT; https://github.com/mw00/project-maya, v1.1.17).
// Ported into Strata (third_party/project-maya/NOTICE.md, LICENSE); changes are in the git history of this file.
// ds4_vision_tower.cu - the vision tower's host side: its weights, and a picture encoded into embedding rows. Its
// kernels are in ds4_vision.cu.
#include "ds4_model_impl.hpp"

namespace strata::ds4 {

bool Model::Impl::read_bf16(const Tensor& t, size_t n, void* dst) {
    if (t.dtype == "BF16") return t.bytes == n * 2 && pk.read(t, dst);
    if (t.dtype != "F32" || t.bytes != n * 4) return false;
    std::vector<float> f(n);
    if (!pk.read(t, f.data())) return false;
    uint16_t* o = (uint16_t*) dst;
    for (size_t i = 0; i < n; ++i) {
        uint32_t u;
        std::memcpy(&u, &f[i], 4);
        o[i] = (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    }
    return true;
}

bool Model::Impl::load_vision(std::string& err) {
    vg.layers = (int) pk.num("vision.num_hidden_layers", 0);
    const int tok = (int) pk.num("image_token_id", -1);
    if (vg.layers <= 0 || tok < 0 || pk.get("vision.patch_embed.proj.weight") == nullptr) return true;   // none
    vg.dim = (int) pk.num("vision.hidden_size", 1024), vg.heads = (int) pk.num("vision.num_attention_heads", 16);
    vg.inter = (int) pk.num("vision.intermediate_size", 2816), vg.patch = (int) pk.num("vision.patch_size", 14);
    vg.merge = (int) pk.num("vision.downsample_ratio", 3), vg.theta = (float) pk.num("vision.rope_theta", 1e4);
    vg.max_tokens = (int) pk.num("vision.max_image_tokens", 1024), vg.out = g.dim;
    const char* mark[3] = {"image_start", "image_newline", "image_end"};
    for (int i = 0; i < 3; ++i) {
        const Tensor* t = pk.get(mark[i]);
        img_mark[i].resize((size_t) g.dim);
        if (t == nullptr || !read_bf16(*t, (size_t) g.dim, img_mark[i].data())) {
            err = std::string(mark[i]) + ": not a BF16 (or F32) row of the model's width";
            return false;
        }
    }
    Dev& d = devs[0];
    // the tower's tensors as BF16: the checkpoint's own, or a GGUF's mmproj (F32 norms / biases / marker rows, which
    // came from BF16, converted back exactly; mlp.w1 as the two files' gate and up, the halves of its rows)
    auto read = [this](const std::string& n, std::vector<int64_t>& shape, void* dst) {
        const Tensor* t = pk.get(n);
        if (t == nullptr && n.size() > 13 && n.compare(n.size() - 13, 13, "mlp.w1.weight") == 0) {
            const std::string pre = n.substr(0, n.size() - 13);
            const Tensor* tg = pk.get(pre + "mlp.gate.weight");
            const Tensor* tu = pk.get(pre + "mlp.up.weight");
            if (tg == nullptr || tu == nullptr || tg->shape.size() != 2 || tg->shape != tu->shape) return false;
            shape = {tg->shape[0] * 2, tg->shape[1]};
            const size_t half = (size_t) tg->shape[0] * (size_t) tg->shape[1];
            return dst == nullptr || (read_bf16(*tg, half, dst) && read_bf16(*tu, half, (uint16_t*) dst + half));
        }
        if (t == nullptr) return false;
        shape = t->shape;
        size_t n_el = 1;
        for (int64_t x : t->shape) n_el *= (size_t) x;
        return dst == nullptr || read_bf16(*t, n_el, dst);
    };
    if (!vt.load(vg, d.id, read, err)) return false;
    img_tok = tok;
    CK(cudaSetDevice(d.id));
    CK(cudaMemGetInfo(&d.free0, &d.total));   // (the split sees what the tower left)
    d.free0 = uma_free(d.id, d.free0);
    return true;
}

bool Model::Impl::encode_image(const uint8_t* rgb, int w, int h, std::vector<uint16_t>& span, std::string& err) {
    if (img_tok < 0) {
        err = "this model has no vision tower (or it is off)";
        return false;
    }
    std::vector<uint16_t> rows;
    int nh = 0, nw = 0;
    if (!vt.encode(rgb, w, h, rows, nh, nw, err)) return false;
    const size_t D = (size_t) g.dim;
    span.resize(((size_t) nh * (nw + 1) + 2) * D);
    uint16_t* o = span.data();
    auto put = [&](const uint16_t* r) {
        std::memcpy(o, r, D * 2);
        o += D;
    };
    put(img_mark[0].data());
    for (int i = 0; i < nh; ++i) {
        for (int j = 0; j < nw; ++j) put(rows.data() + ((size_t) i * nw + j) * D);
        put(img_mark[1].data());
    }
    put(img_mark[2].data());
    return true;
}

const uint16_t* Model::Impl::image_row(int64_t p) const {
    if (imgs.empty()) return nullptr;
    auto it = std::upper_bound(imgs.begin(), imgs.end(), p, [](int64_t v, const Image& im) { return v < im.start; });
    if (it == imgs.begin()) return nullptr;
    const Image& im = *(it - 1);
    const int64_t r = p - im.start;
    if (!im.rows || r >= (int64_t) (im.rows->size() / (size_t) g.dim)) return nullptr;
    return im.rows->data() + (size_t) r * g.dim;
}

bool Model::Impl::vl_bits(int64_t p0, int T, VlBits& b) const {
    if (imgs.empty()) return false;
    std::memset(&b, 0, sizeof b);
    bool any = false;
    for (int t = 0; t < T && t < kVlRows; ++t)
        if (in_image(p0 + t)) {
            b.w[t >> 5] |= 1u << (t & 31);
            any = true;
        }
    return any;
}

}  // namespace strata::ds4
