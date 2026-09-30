# crude saturation model for online test: 1024 in / 256 out, chunked prefill, steady state
GF_tok = 13.958643712e9; GF_head = 1.050673152e9
kv_tok = 131072
def step(Bdec, tp, gemm=700e12, bw=3.0e12, o=0.8e-3, alpha=5e-6, beta=350e9, ctx=1152):
    fin = Bdec/256.0            # requests finishing per step
    P = fin*1024                # prefill tokens per step to keep concurrency constant
    T = Bdec + P
    lin = (T*GF_tok + (Bdec+fin)*GF_head)/tp
    t_lin = max(lin/gemm, (15.01e9/tp)/bw)
    t_att = (Bdec*ctx*kv_tok/tp)/bw + (2*32*32*128*P*1024/tp)/400e12
    ew = (177e3 if tp==1 else 120.5e3)*32*T/bw
    comm = 0 if tp==1 else 65*(alpha + T*4096*2/beta) + 10e-6 + Bdec*64128*2/beta
    t = t_lin + t_att + ew + comm + o + (0 if tp==1 else 0.1e-3)
    return t, (Bdec+fin)/t
for name,B,tp,mult in (("TP1 engine (KV-limited ~330 running)",330,1,1),("DP2 = 2 x TP1",330,1,2),("TP2 (max_num_seqs 512)",512,2,1)):
    t,thr = step(B,tp)
    print(f"{name}: step {t*1e3:.1f} ms, output throughput {mult*thr/1e3:.2f}K tok/s")
for beta in (250e9,150e9):
    t,thr = step(512,2,beta=beta); print(f"TP2 beta={beta/1e9:.0f}GB/s: {thr/1e3:.2f}K tok/s")
