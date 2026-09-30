# D8 a-priori model for Llama-3.1-8B-Instruct on H100 SXM, TP1 vs TP2
V, H, I, L, NH, NKV, HD = 128256, 4096, 14336, 32, 32, 8, 128
q = NH*HD; kv = NKV*HD
attn = H*q + 2*H*kv + q*H          # qkv + o
mlp = 3*H*I                        # gate, up, down
norms_layer = 2*H
per_layer = attn + mlp + norms_layer
embed = V*H; lm_head = V*H; final_norm = H
total = embed + L*per_layer + final_norm + lm_head
print("params: embed", embed, "attn/layer", attn, "mlp/layer", mlp, "per_layer", per_layer)
print("params total", total, " (HF safetensors metadata says 8030261248)")
print("bf16 bytes total", 2*total, "GB", 2*total/1e9, "GiB", 2*total/2**30)
# TP2 per rank: column/row-parallel split, vocab-parallel embed & lm_head (128256 % (64*2) == 0 -> no padding), norms replicated
assert V % 64 == 0 and (V//2) % 64 == 0
tp2_layer = (q//2 + 2*(kv//2))*H + (q//2)*H + 2*(I//2)*H + (I//2)*H + norms_layer
tp2 = L*tp2_layer + (V//2)*H*2 + final_norm
print("TP2 per-rank params", tp2, "bytes", 2*tp2, "GB", 2*tp2/1e9, "GiB", 2*tp2/2**30)
kv_tok = 2*L*NKV*HD*2
print("KV bytes/token total", kv_tok, "TP2 per GPU", kv_tok//2)
mm_no_head = L*(attn+mlp); mm_head = mm_no_head + lm_head
print("matmul params excl lm_head", mm_no_head, "incl", mm_head)
print("FLOPs/token excl lm_head", 2*mm_no_head, "incl", 2*mm_head)
for N in (512, 2048, 8192):
    att = 2*L*NH*HD*N*(N+1)   # causal QK^T + PV (2 FLOPs/MAC), N(N+1)/2 pairs x 2 matmuls
    lin = 2*mm_no_head*N + 2*lm_head   # logits only for last token
    print(f"prefill N={N}: linear {lin/1e12:.3f} TF, attn {att/1e12:.3f} TF, attn share {att/(att+lin):.3%}")
print("decode attn FLOPs/token @ctx c = 4*L*NH*HD*c =", 4*L*NH*HD, "* c")

# ---------------- decode per-step model ----------------
W1 = 2*mm_head + 2*(L*norms_layer+final_norm)   # bytes streamed per step at TP1 (embedding rows negligible)
W2 = W1/2 + (L*norms_layer+final_norm)          # per GPU at TP2 (norms replicated)
print(f"weight bytes streamed/step: TP1 {W1/1e9:.4f} GB, TP2 per GPU {W2/1e9:.4f} GB")
def t_ar(S, alpha, beta): return alpha + S/beta
def decode(B, ctx, bw, o1, o2_extra, alpha, beta, gemm=700e12):
    kv1 = B*ctx*kv_tok
    flops1 = B*(2*mm_head)
    t1_lin = max(W1/bw, flops1/gemm)
    t1 = t1_lin + kv1/bw + o1
    t2_lin = max(W2/bw, flops1/2/gemm)
    ar = 65*t_ar(B*H*2, alpha, beta)
    ag = 10e-6 + B*(V//2)*2/beta
    t2 = t2_lin + (kv1/2)/bw + o1 + o2_extra + ar + ag
    return t1, t2, ar
scen = {
 "central (bw3.0, o0.8, FI/custom AR a=5us)": dict(bw=3.0e12, o1=0.8e-3, o2_extra=0.1e-3, alpha=5e-6, beta=350e9),
 "optimistic (bw3.1, o0.4, a=4us)":           dict(bw=3.1e12, o1=0.4e-3, o2_extra=0.05e-3, alpha=4e-6, beta=400e9),
 "pessimistic (bw2.7, o1.2, a=5us)":          dict(bw=2.7e12, o1=1.2e-3, o2_extra=0.2e-3, alpha=5e-6, beta=300e9),
 "NCCL-forced (bw3.0, o0.8, a=10us)":         dict(bw=3.0e12, o1=0.8e-3, o2_extra=0.1e-3, alpha=10e-6, beta=350e9),
 "NCCL-forced slow (a=17us)":                 dict(bw=3.0e12, o1=0.8e-3, o2_extra=0.1e-3, alpha=17e-6, beta=300e9),
}
for name, p in scen.items():
    print("\n==", name)
    for B in (1, 8, 32, 128):
        t1, t2, ar = decode(B, 1024, **p)
        print(f"  decode B={B:3d} ctx=1024: TP1 {t1*1e3:6.2f} ms  TP2 {t2*1e3:6.2f} ms (AR total {ar*1e3:5.2f} ms)  speedup {t1/t2:4.2f}  eff {t1/t2/2:5.1%}")

# ---------------- prefill model ----------------
def prefill(N, gemm, attn_tf, bw, o, alpha, beta):
    lin = 2*mm_no_head*N + 2*lm_head
    att = 2*L*NH*HD*N*(N+1)
    ew1 = 177e3*L*N          # non-GEMM bytes/token/layer ~177 KB at TP1 (norms, rope, kv write, silu_and_mul)
    ew2 = (64e3 + (177e3-64e3)/2)*L*N   # norms/residual replicated under TP
    t1 = lin/gemm + att/attn_tf + ew1/bw + o
    ar = 65*t_ar(N*H*2, alpha, beta)
    t2 = lin/2/gemm + att/2/attn_tf + ew2/bw + o + 0.1e-3 + ar + (10e-6 + (V//2)*2/beta)
    return t1, t2, ar
for name, p in {"central (GEMM 700, attn 400 TF, a=5us b=350GB/s)": dict(gemm=700e12, attn_tf=400e12, bw=3.0e12, o=0.8e-3, alpha=5e-6, beta=350e9),
                "pessimistic comm (b=250GB/s, a=10us)": dict(gemm=700e12, attn_tf=400e12, bw=3.0e12, o=0.8e-3, alpha=10e-6, beta=250e9),
                "optimistic (GEMM 790, attn 500, b=400)": dict(gemm=790e12, attn_tf=500e12, bw=3.1e12, o=0.5e-3, alpha=4e-6, beta=400e9)}.items():
    print("\n==", name)
    for N in (512, 2048, 8192):
        t1, t2, ar = prefill(N, **p)
        print(f"  prefill N={N:5d}: TP1 {t1*1e3:7.2f} ms  TP2 {t2*1e3:7.2f} ms (AR {ar*1e3:5.2f} ms, msg {N*H*2/2**20:.0f} MiB)  speedup {t1/t2:4.2f}  eff {t1/t2/2:5.1%}")

# ---------------- KV capacity ----------------
GiB = 2**30
total_mem = 79.11*GiB     # torch/cudaMemGetInfo total for "NVIDIA H100 80GB HBM3" (vLLM issues #26833, #27508)
req = total_mem*0.9
w1, w2 = 14.99*GiB, 7.51*GiB   # vLLM "Model loading took" TP1 observed 14.99 GiB; TP2 = 7.48 GiB weights + ~0.03 GiB rope cache
print(f"\nrequested memory @0.9 = {req/GiB:.2f} GiB")
for a1, a2 in ((2.5, 2.9), (3.5, 4.1), (4.82, 5.4), (6.0, 7.0)):
    k1 = req - w1 - a1*GiB; k2 = req - w2 - a2*GiB
    t1 = k1/kv_tok; t2 = k2/(kv_tok/2)
    print(f"  non-KV overhead TP1 {a1} GiB / TP2 {a2} GiB -> KV TP1 {k1/GiB:.2f} GiB = {t1/1e3:.0f}K tok, TP2 {k2/GiB:.2f} GiB/rank = {t2/1e3:.0f}K tok, ratio {t2/t1:.3f}, DP2 total {2*t1/1e3:.0f}K")

print("\n== pessimistic large-message comm (beta=150 GB/s, per scaling-book 58MB/8xH100 figure)")
for N in (512, 2048, 8192):
    t1, t2, ar = prefill(N, gemm=700e12, attn_tf=400e12, bw=3.0e12, o=0.8e-3, alpha=5e-6, beta=150e9)
    print(f"  prefill N={N:5d}: TP1 {t1*1e3:7.2f} ms  TP2 {t2*1e3:7.2f} ms (AR {ar*1e3:5.2f} ms) speedup {t1/t2:4.2f} eff {t1/t2/2:5.1%}")
print("\n== central, mean decode context 1152 (input 1024, 256 outputs)")
for B in (1, 8, 32, 128):
    t1, t2, ar = decode(B, 1152, bw=3.0e12, o1=0.8e-3, o2_extra=0.1e-3, alpha=5e-6, beta=350e9)
    print(f"  decode B={B:3d} ctx=1152: TP1 {t1*1e3:6.2f} ms  TP2 {t2*1e3:6.2f} ms  speedup {t1/t2:4.2f}  eff {t1/t2/2:5.1%}")
# anchor: SGLang measured 158.34 tok/s TP1 bs1 (EAGLE-3 paper Table 4) -> 6.32 ms/token
for bw in (2.7e12, 3.0e12):
    o = 6.32e-3 - (W1 + 1024*kv_tok)/bw
    t2 = (W2 + 1024*kv_tok/2)/bw + o + 0.1e-3 + 65*(5e-6 + 8192/350e9) + 10e-6
    print(f"anchor TP1=6.32ms, bw={bw/1e12}: implied o={o*1e3:.2f} ms -> TP2 {t2*1e3:.2f} ms, speedup {6.32e-3/t2:.2f}")
