


class Counter:
    def __init__(self):
        self.cur_frame = 0
        self.time_step = 1000
        self.block = 0


import os
import math
import torch

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

"""
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema 
"""