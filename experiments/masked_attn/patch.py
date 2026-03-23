"""Asymmetric masked attention for LoRAEdit training.

Background tokens can ONLY attend to background tokens.
Foreground (masked) tokens can attend to ALL tokens.

Implementation: split Q into fg/bg groups, run two flash_attention calls,
merge results. No O(L^2) mask matrix needed.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from typing import Optional

# ---- Global state ----
_mask_frames: Optional[np.ndarray] = None   # [F, H, W] float32 in {0, 1}
_token_mask: Optional[torch.Tensor] = None  # [L] bool, True = foreground
_cached_grid_key: Optional[tuple] = None    # for cache invalidation


def load_mask(data_dir: str) -> None:
    """Load mask frames from source_masks/ directory."""
    global _mask_frames

    mask_dir = os.path.join(data_dir, "source_masks")
    if os.path.isdir(mask_dir):
        files = sorted(f for f in os.listdir(mask_dir)
                       if f.endswith(('.png', '.jpg')))
        masks = []
        for f in files:
            img = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
            masks.append((img > 127).astype(np.float32))
        _mask_frames = np.stack(masks)
    else:
        cap = cv2.VideoCapture(os.path.join(data_dir, "inference_mask.mp4"))
        masks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masks.append((gray > 127).astype(np.float32))
        cap.release()
        _mask_frames = np.stack(masks)

    fg_ratio = _mask_frames.mean()
    print(f"[MaskedAttn] Loaded mask: shape={_mask_frames.shape}, "
          f"foreground={fg_ratio:.3f}")


def _compute_token_mask(grid_sizes: torch.Tensor) -> torch.Tensor:
    """Convert spatial mask [F,H,W] → token mask [L] using grid_sizes."""
    global _token_mask, _cached_grid_key
    assert _mask_frames is not None, "Call load_mask() first"

    f_grid, h_grid, w_grid = [int(x) for x in grid_sizes[0].tolist()]
    grid_key = (f_grid, h_grid, w_grid)

    if _cached_grid_key == grid_key and _token_mask is not None:
        return _token_mask

    f_orig, h_orig, w_orig = _mask_frames.shape
    mask_t = torch.from_numpy(_mask_frames).float()  # [F, H, W]

    # Temporal: nearest-neighbor resample
    if f_grid != f_orig:
        indices = torch.linspace(0, f_orig - 1, f_grid).long()
        mask_t = mask_t[indices]

    # Spatial: resize each frame
    mask_t = mask_t.unsqueeze(1)  # [F, 1, H, W]
    mask_t = F.interpolate(mask_t, size=(h_grid, w_grid), mode='nearest')
    mask_t = mask_t.squeeze(1)  # [F_grid, H_grid, W_grid]

    _token_mask = (mask_t.flatten() > 0.5)
    _cached_grid_key = grid_key

    n_fg = _token_mask.sum().item()
    n_total = _token_mask.numel()
    print(f"[MaskedAttn] Token mask: grid=({f_grid},{h_grid},{w_grid}), "
          f"tokens={n_total}, fg={n_fg} ({n_fg/n_total:.1%})")
    return _token_mask


def apply_patch() -> None:
    """Monkey-patch WanSelfAttention.forward with asymmetric masked attention."""
    import sys
    loraedit_dir = '/home/users/ntu/song0304/code/LoRAEdit'
    if loraedit_dir not in sys.path:
        sys.path.insert(0, loraedit_dir)

    from submodules.Wan2_1.wan.modules.model import WanSelfAttention, rope_apply
    from submodules.Wan2_1.wan.modules.attention import flash_attention

    _original_forward = WanSelfAttention.forward

    def masked_forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # QKV projection (same as original)
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # Apply RoPE (same as original)
        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        # Get token mask
        token_mask = _compute_token_mask(grid_sizes)
        device = q_rope.device
        mask = token_mask.to(device)

        fg_idx = mask.nonzero(as_tuple=True)[0]
        bg_idx = (~mask).nonzero(as_tuple=True)[0]

        # Pre-allocate output [B, seq_len_padded, H, D]
        output = q_rope.new_zeros(q_rope.shape)

        # Foreground queries → attend to ALL keys/values
        if fg_idx.numel() > 0:
            q_fg = q_rope[:, fg_idx]  # [B, L_fg, H, D]
            out_fg = flash_attention(
                q=q_fg, k=k_rope, v=v,
                k_lens=seq_lens,
                window_size=self.window_size)
            output[:, fg_idx] = out_fg

        # Background queries → attend to BACKGROUND keys/values only
        if bg_idx.numel() > 0:
            q_bg = q_rope[:, bg_idx]  # [B, L_bg, H, D]
            k_bg = k_rope[:, bg_idx]  # [B, L_bg, H, D]
            v_bg = v[:, bg_idx]       # [B, L_bg, H, D]
            out_bg = flash_attention(
                q=q_bg, k=k_bg, v=v_bg,
                window_size=self.window_size)
            output[:, bg_idx] = out_bg

        x = output.flatten(2)
        x = self.o(x)
        return x

    WanSelfAttention.forward = masked_forward
    print("[MaskedAttn] Patched WanSelfAttention.forward (asymmetric masked attention)")
