


class Counter:
    def __init__(self):
        self.cur_frame = 0
        self.time_step = 1000
        self.block = 0


import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

def attn_map(counter, q, k, v, *, out_root="attn_vis", chunk_k=1024, eps=1e-12):
    """
    Save per-head attention maps for:
      current frame queries (Q tokens) attending to cached keys (K tokens),
      aggregated by mean over Q (so one map per history frame).

    Args:
        counter: Counter with fields {cur_frame, time_step, block}
        q, k, v: torch.Tensor
            Expected shapes:
              q: [B, Q, H, D]
              k: [B, K, H, D]
              v: [B, K, H, D]   (not needed for visualization; kept for API symmetry)
        out_root: output root folder
        chunk_k: key-token chunk size for streaming softmax
    """
    # -----------------------------
    # 0) sanitize shapes
    # -----------------------------
    if q.dim() == 3:  # [Q,H,D]
        q = q.unsqueeze(0)
    if k.dim() == 3:
        k = k.unsqueeze(0)
    if v is not None and v.dim() == 3:
        v = v.unsqueeze(0)

    assert q.dim() == 4 and k.dim() == 4, f"Expect q,k to be 4D. Got q={q.shape}, k={k.shape}"
    B, Q, H, D = q.shape
    B2, K, H2, D2 = k.shape
    assert B == B2 and H == H2 and D == D2, f"Shape mismatch: q={q.shape}, k={k.shape}"

    # -----------------------------
    # 1) infer latent grid (H_lat, W_lat) from Q
    #    (assumes Q == frame_seqlen == H_lat * W_lat)
    # -----------------------------
    def infer_hw(n: int):
        # Prefer exact known common case
        if n == 1560:
            return 30, 52  # common Wan latent grid
        # Find factor pair closest to square
        r = int(math.sqrt(n))
        best = None
        for a in range(r, 0, -1):
            if n % a == 0:
                b = n // a
                best = (a, b)
                break
        if best is None:
            return 1, n
        h_lat, w_lat = best
        # enforce w>=h for consistency
        return (h_lat, w_lat) if w_lat >= h_lat else (w_lat, h_lat)

    H_lat, W_lat = infer_hw(Q)
    # print(f'H_lat={H_lat}, W_lat={W_lat}, q={q.shape}')

    frame_seqlen = H_lat * W_lat
    if frame_seqlen != Q:
        # If you ever process multiple frames at once (Q = n_frames * frame_seqlen),
        # you should split Q first. Here we keep it strict to match your current goal.
        raise ValueError(
            f"Q={Q} is not a single-frame grid (inferred {H_lat}x{W_lat}={frame_seqlen}). "
            f"If Q includes multiple frames, split q by frames before calling attn_map()."
        )

    # K should be multiple of frame_seqlen in your cache logic; handle remainder defensively.
    num_hist_frames = K // frame_seqlen
    usable_K = num_hist_frames * frame_seqlen
    if usable_K == 0:
        return None

    # Only visualize full frames worth of keys
    if usable_K != K:
        k = k[:, :usable_K]
        K = usable_K

    # -----------------------------
    # 2) streaming softmax to get attention weights (mean over Q)
    #    We compute:
    #      A[b,h,q,k] = softmax_k( (Q·K)/sqrt(D) )
    #      acc[b,h,k] = mean_q A[b,h,q,k]
    # -----------------------------
    with torch.no_grad():
        # [B,H,Q,D] and [B,H,K,D]
        Qh = q.permute(0, 2, 1, 3).contiguous()
        Kh = k.permute(0, 2, 1, 3).contiguous()

        scale = 1.0 / math.sqrt(D)

        # First pass: compute logZ per (B,H,Q) via streaming logsumexp
        m = torch.full((B, H, Q), float("-inf"), device=q.device, dtype=torch.float32)
        s = torch.zeros((B, H, Q), device=q.device, dtype=torch.float32)

        for start in range(0, K, chunk_k):
            end = min(start + chunk_k, K)
            Khc = Kh[:, :, start:end, :].to(torch.float32)  # [B,H,Kc,D]
            Qhc = Qh.to(torch.float32)                      # [B,H,Q,D]

            logits = torch.einsum("bhqd,bhkd->bhqk", Qhc, Khc) * scale  # [B,H,Q,Kc]
            max_chunk = logits.max(dim=-1).values  # [B,H,Q]

            m_new = torch.maximum(m, max_chunk)
            # update s with rescaling to keep stability
            s = s * torch.exp(m - m_new) + torch.exp(logits - m_new.unsqueeze(-1)).sum(dim=-1)
            m = m_new

        logZ = m + torch.log(s + eps)  # [B,H,Q]

        # Second pass: accumulate mean over Q for each key token
        acc = torch.zeros((B, H, K), device=q.device, dtype=torch.float32)

        for start in range(0, K, chunk_k):
            end = min(start + chunk_k, K)
            Khc = Kh[:, :, start:end, :].to(torch.float32)  # [B,H,Kc,D]
            Qhc = Qh.to(torch.float32)                      # [B,H,Q,D]

            logits = torch.einsum("bhqd,bhkd->bhqk", Qhc, Khc) * scale  # [B,H,Q,Kc]
            w = torch.exp(logits - logZ.unsqueeze(-1))                   # [B,H,Q,Kc]
            acc[:, :, start:end] = w.mean(dim=2)                         # mean over Q -> [B,H,Kc]

        # Move to CPU for saving
        acc_cpu = acc.cpu()

    # -----------------------------
    # 3) save images: /{cur_frame}/{time_step}/{block}/{head}/frame{i}.png
    # -----------------------------
    try:
        from PIL import Image
    except ImportError:
        Image = None

    for b in range(B):
        for h in range(H):
            head_dir = os.path.join(
                out_root,
                f'cur_frame{str(counter.cur_frame)}',
                f't{str(counter.time_step)}',
                f'b{str(counter.block)}',
                f"h{str(h)}",
            )
            os.makedirs(head_dir, exist_ok=True)

            # You can choose normalization strategy:
            # (a) per-frame normalization (default): visually clear, but loses absolute comparability
            # (b) per-head global normalization across all frames: comparable, but might look "washed out"
            head_vec = acc_cpu[b, h]  # [K]

            for fi in range(num_hist_frames):
                seg = head_vec[fi * frame_seqlen:(fi + 1) * frame_seqlen].view(H_lat, W_lat)

                # per-frame normalize to [0,255]
                mn = float(seg.min())
                mx = float(seg.max())
                denom = (mx - mn) if (mx - mn) > 1e-20 else 1.0
                img = ((seg - mn) / denom * 255.0).clamp(0, 255).to(torch.uint8).numpy()

                out_path = os.path.join(head_dir, f"frame{fi}.png")
                if Image is not None:
                    Image.fromarray(img, mode="L").save(out_path)
                else:
                    # fallback if PIL isn't available
                    import imageio.v2 as imageio
                    imageio.imwrite(out_path, img)

    return None


def _to_bhnd(x: torch.Tensor) -> torch.Tensor:
    """
    Try to coerce x into [B, H, N, D].
    Common cases:
      - [B, H, N, D] (already)
      - [B, N, H, D]
      - [H, N, D]   (no batch)
      - [N, H, D]   (no batch)
    """
    if x.dim() == 4:
        B, a, b, D = x.shape
        # Heuristic: head dim usually <= 32 and token dim usually larger
        if a <= 64 and b > a:
            # [B, H, N, D]
            return x
        if b <= 64 and a > b:
            # [B, N, H, D] -> [B, H, N, D]
            return x.permute(0, 2, 1, 3).contiguous()
        # Fallback: assume [B, H, N, D]
        return x

    if x.dim() == 3:
        a, b, D = x.shape
        # [H, N, D]
        if a <= 64 and b > a:
            return x.unsqueeze(0)  # [1, H, N, D]
        # [N, H, D]
        if b <= 64 and a > b:
            return x.permute(1, 0, 2).unsqueeze(0).contiguous()
        # Fallback: treat first as H
        return x.unsqueeze(0)

    raise ValueError(f"Unsupported tensor rank: {x.dim()} for shape {tuple(x.shape)}")


"""
deep forcing style attention map
"""
@torch.no_grad()
def line_attn_map(counter, q, k, v,
             tokens_per_frame: int = 1560,
             out_root: str = ".",
             exclude_query_frames_from_keys: bool = True,
             max_query_tokens: int | None = None):
    """
    Reproduce Deep Forcing-style "attention distribution across earlier frames":
      For each head: average attention *logits* from all query tokens to each key-frame's tokens,
      producing a 1D curve over key frames.

    Args:
      counter: has .cur_frame, .time_step, .block
      q,k,v: projected tensors (before softmax attention), any of the supported shapes
      tokens_per_frame: paper uses 1560 tokens/frame in latent patch layout
      exclude_query_frames_from_keys: if k contains the same chunk's tokens, drop the last q_frames from keys
      max_query_tokens: optional subsample of query tokens for speed (strided)
    """
    if counter.cur_frame < 19:
        return

    if counter.time_step > 625:
        return

    print(f"frame:{counter.cur_frame}, time:{counter.time_step}, block:{counter.block}")

    q = _to_bhnd(q)
    k = _to_bhnd(k)

    device = q.device
    q = q.float()
    k = k.float()

    B, H, Q, D = q.shape
    _, _, K, _ = k.shape

    # Optional query subsample (keeps plot stable but faster)
    if max_query_tokens is not None and Q > max_query_tokens:
        idx = torch.linspace(0, Q - 1, steps=max_query_tokens, device=device).long()
        q = q[:, :, idx, :]
        Q = q.shape[2]

    # Estimate how many frames are inside q (only used to exclude keys)
    q_frames = max(1, Q // tokens_per_frame) if (Q % tokens_per_frame == 0) else 1
    k_frames_total = max(1, math.ceil(K / tokens_per_frame))

    # Optionally exclude "current chunk" keys so the curve is only over earlier frames (like the paper fig)
    if exclude_query_frames_from_keys and k_frames_total > q_frames:
        keep_frames = k_frames_total - q_frames
        K_use = min(K, keep_frames * tokens_per_frame)
    else:
        K_use = K

    # F = max(1, math.ceil(K_use / tokens_per_frame))
    scale = 1.0 / math.sqrt(D)

    # Prepare output directory: ./cur_frame/block/hX.png
    out_dir = os.path.join(out_root, str(counter.cur_frame), str(counter.block))
    os.makedirs(out_dir, exist_ok=True)

    # Precompute per-head q_sum to avoid huge QxK matmuls:
    # sum_{q,k} (q·k) = (sum_q q) · (sum_k k)  (per batch), then sum over batch.
    # This reproduces mean of pre-softmax logits over all (query token, key token) pairs in a frame.
    q_sum = q.sum(dim=2)  # [B, H, D]

    # Loop heads and frames
    for h in range(H):
        # y 的长度是 K_use：每个 key token 一个点
        y = np.empty(K_use, dtype=np.float32)

        qh_sum = q_sum[:, h, :]  # [B, D]

        # 分块算，避免一次性吃太多显存
        chunk = 8192
        for s in range(0, K_use, chunk):
            e = min(K_use, s + chunk)

            # kh: [B, Tk, D]
            kh = k[:, h, s:e, :]

            # 对每个 key token j：score_j = mean_{b,q} (q·k_j)/sqrt(D)
            # 利用 sum_q：mean_{q}(q·k) = (sum_q q)·k / Q
            # scores_bj = (qh_sum_b · kh_bj) / (Q*sqrt(D))  -> [B, Tk]
            scores = (kh * qh_sum[:, None, :]).sum(dim=-1) * (scale / max(Q, 1))

            # 再对 batch 平均：-> [Tk]
            scores = scores.mean(dim=0)

            y[s:e] = scores.detach().float().cpu().numpy()

        # Plot (line + filled area), matching the paper vibe
        x = np.arange(K_use)
        plt.figure(figsize=(6.0, 3.2))
        plt.plot(x, y, linewidth=0.5)
        plt.fill_between(x, y, alpha=0.25)
        plt.xlabel("Key Tokens (flattened by time)")
        plt.ylabel("Query-avg Attn Logit (per token)")
        plt.title(f"L{counter.block} H{h}  (t={counter.time_step}, cur={counter.cur_frame})")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"h{h}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    # 这个函数通常只做 side-effect 可视化，不改变前向
    return None

"""
deep forcing style,
attention score graph
"""
@torch.no_grad()
def line_attn_score(counter, q, k, v,
             tokens_per_frame: int = 1560,
             out_root: str = ".",
             exclude_query_frames_from_keys: bool = True,
             max_query_tokens: int | None = None):
    """
    Reproduce Deep Forcing-style "attention distribution across earlier frames":
      For each head: average attention *logits* from all query tokens to each key-frame's tokens,
      producing a 1D curve over key frames.

    Args:
      counter: has .cur_frame, .time_step, .block
      q,k,v: projected tensors (before softmax attention), any of the supported shapes
      tokens_per_frame: paper uses 1560 tokens/frame in latent patch layout
      exclude_query_frames_from_keys: if k contains the same chunk's tokens, drop the last q_frames from keys
      max_query_tokens: optional subsample of query tokens for speed (strided)
    """
    if counter.cur_frame < 21:
        return

    if counter.time_step > 625:
        return

    print(f"frame:{counter.cur_frame}, time:{counter.time_step}, block:{counter.block}")

    q = _to_bhnd(q)
    k = _to_bhnd(k)

    device = q.device
    q = q.float()
    k = k.float()

    B, H, Q, D = q.shape
    _, _, K, _ = k.shape

    # Optional query subsample (keeps plot stable but faster)
    if max_query_tokens is not None and Q > max_query_tokens:
        idx = torch.linspace(0, Q - 1, steps=max_query_tokens, device=device).long()
        q = q[:, :, idx, :]
        Q = q.shape[2]

    # Estimate how many frames are inside q (only used to exclude keys)
    q_frames = max(1, Q // tokens_per_frame) if (Q % tokens_per_frame == 0) else 1
    k_frames_total = max(1, math.ceil(K / tokens_per_frame))

    # Optionally exclude "current chunk" keys so the curve is only over earlier frames (like the paper fig)
    if exclude_query_frames_from_keys and k_frames_total > q_frames:
        keep_frames = k_frames_total - q_frames
        K_use = min(K, keep_frames * tokens_per_frame)
    else:
        K_use = K

    # F = max(1, math.ceil(K_use / tokens_per_frame))
    scale = 1.0 / math.sqrt(D)

    # Prepare output directory: ./cur_frame/block/hX.png
    out_dir = os.path.join(out_root, str(counter.cur_frame), str(counter.block))
    os.makedirs(out_dir, exist_ok=True)

    # Precompute per-head q_sum to avoid huge QxK matmuls:
    # sum_{q,k} (q·k) = (sum_q q) · (sum_k k)  (per batch), then sum over batch.
    # This reproduces mean of pre-softmax logits over all (query token, key token) pairs in a frame.
    q_sum = q.sum(dim=2)  # [B, H, D]

    # Loop heads
    y_sum = np.zeros(K_use, dtype=np.float32)

    def plot_y(x, y, h):
        # Plot (line + filled area), matching the paper vibe
        plt.figure(figsize=(6.0, 3.2))
        plt.plot(x, y, linewidth=0.5)
        plt.fill_between(x, y, alpha=0.25)
        plt.xlabel("Key Tokens (flattened by time)")
        plt.ylabel("Query-avg Attn Logit (per token)")
        plt.title(f"L{counter.block} H{h}  (t={counter.time_step}, cur={counter.cur_frame})")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"h{h}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

    for h in range(H):
        y = np.empty(K_use, dtype=np.float32)

        qh = q[:, h, :, :]  # [B, Q, D]

        chunk = 8192

        # ---- Pass 1: global max over ALL keys (for stable softmax) ----
        max_logit = torch.full((B, Q), -float("inf"), device=device, dtype=qh.dtype)
        for s in range(0, K_use, chunk):
            e = min(K_use, s + chunk)
            kh = k[:, h, s:e, :]  # [B, Tk, D]

            # logits: [B, Q, Tk]
            logits = torch.einsum("bqd,bkd->bqk", qh, kh) * scale
            max_logit = torch.maximum(max_logit, logits.max(dim=-1).values)

        # ---- Pass 2: global denom over ALL keys ----
        denom = torch.zeros((B, Q), device=device, dtype=qh.dtype)
        for s in range(0, K_use, chunk):
            e = min(K_use, s + chunk)
            kh = k[:, h, s:e, :]  # [B, Tk, D]
            logits = torch.einsum("bqd,bkd->bqk", qh, kh) * scale
            denom = denom + torch.exp(logits - max_logit.unsqueeze(-1)).sum(dim=-1)

        # Avoid divide-by-zero (shouldn't happen, but safe)
        denom = denom.clamp_min(1e-12)

        # ---- Pass 3: compute weights + "vote" per key token ----
        for s in range(0, K_use, chunk):
            e = min(K_use, s + chunk)
            kh = k[:, h, s:e, :]  # [B, Tk, D]
            logits = torch.einsum("bqd,bkd->bqk", qh, kh) * scale

            # softmax weights: [B, Q, Tk]
            w = torch.exp(logits - max_logit.unsqueeze(-1)) / denom.unsqueeze(-1)

            # "vote score" per key token = sum over queries (current frame tokens)
            # -> [B, Tk]
            vote = w.sum(dim=1)

            # optional: if you want query-AVERAGE vote instead of sum, use:
            # vote = w.mean(dim=1)

            # batch average -> [Tk]
            vote = vote.mean(dim=0)
            y[s:e] = vote.detach().float().cpu().numpy()

        y_sum = y_sum + y
        plot_y(np.arange(K_use), y, h)

    plot_y(np.arange(K_use), y_sum, 12)

    # 这个函数通常只做 side-effect 可视化，不改变前向
    return None


def _to_bhqd(x: torch.Tensor) -> torch.Tensor:
    """
    Try to normalize q/k/v into shape [B, H, S, D].
    Common cases:
      - [B, H, S, D]
      - [B, S, H, D]
    """
    if x is None:
        return None
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor for q/k/v, got shape={tuple(x.shape)}")

    B, A, B_or_S, D = x.shape
    # Heuristic: head count is usually <= 128, sequence length usually much larger.
    if A <= 128 and B_or_S > A:
        # [B, H, S, D]
        return x
    if B_or_S <= 128 and A > B_or_S:
        # [B, S, H, D] -> [B, H, S, D]
        return x.permute(0, 2, 1, 3).contiguous()

    # Fallback: assume already [B,H,S,D]
    return x


@torch.no_grad()
def graph_attn_score(counter, q, k, v):
    # ===== keep your switches =====
    if counter.cur_frame < 21:
        return
    if counter.time_step > 625:
        return

    # ===== output root (you can override via env var) =====
    out_root = os.environ.get("ATTN_SCORE_ROOT", "attn_score_fig2")
    frame_dir = os.path.join(out_root, f"t{int(counter.time_step):04d}/frame_{int(counter.cur_frame):04d}/layer{int(counter.block):02d}")
    os.makedirs(frame_dir, exist_ok=True)

    # ===== normalize shapes =====
    q = _to_bhqd(q)
    k = _to_bhqd(k)

    # pick first sample in batch
    qb = q[0]  # [H, Q, D]
    kb = k[0]  # [H, K, D]

    H, Q, D = qb.shape
    _, K, _ = kb.shape

    # evenly spaced indices (deterministic,方便复现)
    q_idx = torch.arange(Q, device=qb.device)
    k_idx = torch.arange(K, device=kb.device)

    # ===== plot per-head heatmap =====
    # IMPORTANT: do imports here so this function doesn't force matplotlib on every run unless it triggers
    import numpy as np
    import matplotlib.pyplot as plt

    scale = 1.0 / math.sqrt(D)

    for h in range(H):
        qh = qb[h].index_select(0, q_idx).float()  # [qn, D]
        kh = kb[h].index_select(0, k_idx).float()  # [kn, D]

        # attention logits + softmax over keys
        logits = (qh @ kh.transpose(0, 1)) * scale          # [qn, kn]
        attn = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        # Make the visualization more “pattern-revealing”
        # (optional) clamp tiny values for contrast; keep mild so it doesn’t lie.
        # attn = np.clip(attn, 0.0, np.quantile(attn, 0.999))

        plt.figure(figsize=(10, 4))
        # robust color scaling via quantiles
        vmin = np.quantile(attn, 0.05)
        vmax = np.quantile(attn, 0.995)
        attn_vis = np.power(attn, 0.995)  # sqrt stretch

        plt.imshow(attn_vis,
                   aspect="auto",
                   interpolation="nearest",
                   cmap="cividis",
                    vmin = vmin,
                    vmax = vmax,
                )
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(
            f"Attn Pattern | frame={counter.cur_frame} | t={counter.time_step} | layer={counter.block} | head={h}"
        )
        plt.xlabel("Key token index (sampled)")
        plt.ylabel("Query token index (sampled)")

        save_path = os.path.join(
            frame_dir,
            f"h{h:02d}.png"
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()


"""
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema 
"""