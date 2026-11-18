"""mini-pixel-art: standalone inference for a tiny 16x16 pixel-art text-to-image DiT.

Depends only on torch, numpy, safetensors and Pillow. One file, no training code.

    python mini_pixel_art.py --model m-p1 "red potion bottle" "side-view knight with sword" --out sheet.png

Files expected next to this script (or under --root):
    <model>/model.safetensors, <model>/config.json, vocab.json, palette.bin
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, NULL = 0, 1, 2
TEXT_LEN = 12
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


# ---------------------------------------------------------------- text
def words(caption: str) -> list[str]:
    return _WORD.findall(caption.lower())


def encode_caption(caption: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(w, UNK) for w in words(caption)][:TEXT_LEN]
    return ids + [PAD] * (TEXT_LEN - len(ids))


# ---------------------------------------------------------------- model
def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")
    return torch.cat([f(p.reshape(-1, 1) * omega) for p in (ys, xs) for f in (torch.sin, torch.cos)], 1)


def time_features(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float().unsqueeze(-1) * 1000.0 * freq
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * mlp_ratio)
        self.fc2 = nn.Linear(width * mlp_ratio, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.relu(self.fc1(self.norm2(x))))


class TinyDiT(nn.Module):
    """Single-stream pixel-space DiT: [time token, 12 text tokens, 256/patch² pixel tokens]. Predicts the clean image."""

    def __init__(self, n_vocab: int, width: int, depth: int, heads: int, patch: int, mlp_ratio: int, text_dim: int, **_):
        super().__init__()
        self.patch, self.grid = patch, 16 // patch
        pdim = 4 * patch * patch
        self.width = width
        self.text_emb = nn.Embedding(n_vocab, text_dim)
        self.text_proj = nn.Linear(text_dim, width)
        self.text_pos = nn.Parameter(torch.zeros(TEXT_LEN, width))
        self.time_mlp = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))])
        self.pix_in = nn.Linear(pdim, width)
        self.register_buffer("pix_pos", sincos_2d(self.grid, width), persistent=False)
        self.blocks = nn.ModuleList(Block(width, heads, mlp_ratio) for _ in range(depth))
        self.norm_out = nn.LayerNorm(width)
        self.pix_out = nn.Linear(width, pdim)

    def forward(self, z: torch.Tensor, t: torch.Tensor, cap_ids: torch.Tensor) -> torch.Tensor:
        b, p, g = z.shape[0], self.patch, self.grid
        time_tok = self.time_mlp[0](time_features(t, self.width)).unsqueeze(1)
        txt = self.text_proj(self.text_emb(cap_ids)) + self.text_pos
        pix = self.pix_in(z.view(b, 4, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(b, g * g, 4 * p * p)) + self.pix_pos
        x = torch.cat([time_tok, txt, pix], 1)
        for blk in self.blocks:
            x = blk(x)
        y = self.pix_out(self.norm_out(x[:, 1 + TEXT_LEN:]))
        return y.view(b, g, g, 4, p, p).permute(0, 3, 1, 4, 2, 5).reshape(b, 4, 16, 16)


# ---------------------------------------------------------------- sampling
@torch.no_grad()
def sample(model: TinyDiT, cap_ids: torch.Tensor, noise: torch.Tensor, steps: int = 16, cfg: float = 1.5,
           cfg_alpha: float = 1.5, t_stop: float = 0.05) -> torch.Tensor:
    """Flow matching z_t = (1-t)x + t·eps, Euler from t=1 to 0 with classifier-free guidance.

    The grid stops at t_stop and the last step jumps straight to x̂: the released models were trained with
    max(t, 0.05) in the loss and their predictions are biased below t = 0.05. Network calls = steps (×2 for CFG).
    """
    null = torch.full_like(cap_ids, PAD)
    null[:, 0] = NULL
    scale = torch.tensor([cfg, cfg, cfg, cfg_alpha], device=noise.device).view(1, 4, 1, 1)
    ts = torch.linspace(1.0, t_stop, steps).tolist() + [0.0] if steps > 1 else [1.0, 0.0]
    z = noise.clone()
    for t, t_next in zip(ts[:-1], ts[1:]):
        tt = torch.full((2 * z.shape[0],), t, device=z.device)
        xc, xu = model(torch.cat([z, z]), tt, torch.cat([cap_ids, null])).float().chunk(2)
        x0 = xu + scale * (xc - xu)
        z = (t_next / t) * z + (1 - t_next / t) * x0
    return z


def snap(x: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """[B,4,16,16] in [-1,1] -> palette indices [B,256]; 0 = transparent (alpha < 0), 1..4096 = palette colour."""
    px = x.float().permute(0, 2, 3, 1).reshape(-1, 4)
    pal = palette[1:].float() / 127.5 - 1.0
    c = px[:, :3]
    d = (c * c).sum(-1, keepdim=True) - 2.0 * c @ pal.T + (pal * pal).sum(-1)
    out = d.argmin(-1) + 1
    out[px[:, 3] < 0] = 0
    return out.view(-1, 256)


def to_rgba(tokens: torch.Tensor, palette: torch.Tensor) -> np.ndarray:
    """[B,256] -> [B,16,16,4] uint8."""
    tok = tokens.long().cpu()
    rgb = palette.cpu()[tok]
    a = ((tok != 0).to(torch.uint8) * 255).unsqueeze(-1)
    return torch.cat([rgb, a], -1).view(-1, 16, 16, 4).numpy()


# ---------------------------------------------------------------- loading
class Pipeline:
    def __init__(self, root: str | Path, model: str = "m-p1", device: str = "cpu"):
        from safetensors.torch import load_file
        root = Path(root)
        cfg = json.loads((root / model / "config.json").read_text())
        self.vocab = json.loads((root / "vocab.json").read_text())
        self.palette = torch.from_numpy(np.frombuffer((root / "palette.bin").read_bytes(), np.uint8).reshape(-1, 3).copy()).to(device)
        self.model = TinyDiT(**cfg["model"]).to(device).eval()
        self.model.load_state_dict(load_file(str(root / model / "model.safetensors"), device=device))
        self.sampler = cfg["sampler"]
        self.device = device

    def unknown_words(self, captions: list[str]) -> list[str]:
        return [w for c in captions for w in words(c) if w not in self.vocab]

    def __call__(self, captions: list[str], seed: int = 0, **kw) -> np.ndarray:
        """Returns [len(captions),16,16,4] uint8 RGBA. kw overrides steps / cfg / cfg_alpha / t_stop."""
        ids = torch.tensor([encode_caption(c, self.vocab) for c in captions], device=self.device)
        noise = torch.randn(len(captions), 4, 16, 16, generator=torch.Generator().manual_seed(seed)).to(self.device)
        z = sample(self.model, ids, noise, **{**self.sampler, **kw})
        return to_rgba(snap(z, self.palette), self.palette)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captions", nargs="+")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--model", default="m-p1")
    ap.add_argument("--out", default="sheet.png")
    ap.add_argument("--n", type=int, default=8, help="images per caption")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--cfg", type=float)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=8, help="nearest-neighbour upscale of the saved sheet")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from PIL import Image
    pipe = Pipeline(args.root, args.model, args.device)
    unk = pipe.unknown_words(args.captions)
    if unk:
        print("out-of-vocabulary words (ignored as <unk>):", unk)
    kw = {k: v for k, v in (("steps", args.steps), ("cfg", args.cfg)) if v is not None}
    imgs = pipe([c for c in args.captions for _ in range(args.n)], seed=args.seed, **kw)
    s = args.scale
    sheet = Image.new("RGBA", (args.n * 16 * s, len(args.captions) * 16 * s), (0, 0, 0, 0))
    for k, im in enumerate(imgs):
        sheet.alpha_composite(Image.fromarray(im, "RGBA").resize((16 * s, 16 * s), Image.NEAREST),
                              ((k % args.n) * 16 * s, (k // args.n) * 16 * s))
    sheet.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
