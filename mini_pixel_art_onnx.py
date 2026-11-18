"""mini-pixel-art with ONNX Runtime: no PyTorch needed (numpy + onnxruntime + Pillow).

    python mini_pixel_art_onnx.py --model m-p1 "red potion bottle" "golden key" --out sheet.png

<model>/model.onnx takes z [B,4,16,16] float32, t [B] float32 (1 = pure noise, 0 = clean) and
cap_ids [B,12] int64, and returns the predicted clean image x [B,4,16,16] in [-1,1]. The sampling loop below
is the same as in mini_pixel_art.py. Noise comes from numpy here, so images differ from the PyTorch script
for the same seed; pass your own noise to compare.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

PAD, UNK, NULL = 0, 1, 2
TEXT_LEN = 12
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def words(caption: str) -> list[str]:
    return _WORD.findall(caption.lower())


def encode_caption(caption: str, vocab: dict[str, int]) -> list[int]:
    ids = [vocab.get(w, UNK) for w in words(caption)][:TEXT_LEN]
    return ids + [PAD] * (TEXT_LEN - len(ids))


class OnnxPipeline:
    def __init__(self, root: str | Path, model: str = "m-p1", providers: list[str] | None = None):
        import onnxruntime as ort
        root = Path(root)
        self.vocab = json.loads((root / "vocab.json").read_text())
        self.palette = np.frombuffer((root / "palette.bin").read_bytes(), np.uint8).reshape(-1, 3)
        self.sampler = json.loads((root / model / "config.json").read_text())["sampler"]
        self.sess = ort.InferenceSession(str(root / model / "model.onnx"), providers=providers or ["CPUExecutionProvider"])
        self._pal = self.palette[1:].astype(np.float32) / 127.5 - 1.0

    def predict(self, z: np.ndarray, t: float, ids: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"z": z.astype(np.float32), "t": np.full(len(z), t, np.float32), "cap_ids": ids.astype(np.int64)})[0]

    def sample(self, ids: np.ndarray, noise: np.ndarray, steps: int = 16, cfg: float = 1.5, cfg_alpha: float = 1.5,
               t_stop: float = 0.05) -> np.ndarray:
        """Euler from t=1 to 0 with classifier-free guidance; the grid stops at t_stop and the last step jumps to x̂."""
        null = np.full_like(ids, PAD)
        null[:, 0] = NULL
        scale = np.array([cfg, cfg, cfg, cfg_alpha], np.float32).reshape(1, 4, 1, 1)
        ts = list(np.linspace(1.0, t_stop, steps)) + [0.0] if steps > 1 else [1.0, 0.0]
        z = noise.astype(np.float32).copy()
        for t, t_next in zip(ts[:-1], ts[1:]):
            xc, xu = np.split(self.predict(np.concatenate([z, z]), float(t), np.concatenate([ids, null])), 2)
            x0 = xu + scale * (xc - xu)
            r = np.float32(t_next / t)
            z = r * z + (1 - r) * x0
        return z

    def snap(self, x: np.ndarray) -> np.ndarray:
        """[B,4,16,16] -> palette indices [B,256]; 0 = transparent."""
        px = x.transpose(0, 2, 3, 1).reshape(-1, 4)
        c = px[:, :3]
        d = (c * c).sum(-1, keepdims=True) - 2.0 * c @ self._pal.T + (self._pal * self._pal).sum(-1)
        out = d.argmin(-1) + 1
        out[px[:, 3] < 0] = 0
        return out.reshape(-1, 256)

    def to_rgba(self, tokens: np.ndarray) -> np.ndarray:
        rgb = self.palette[tokens]
        a = ((tokens != 0) * 255).astype(np.uint8)[..., None]
        return np.concatenate([rgb, a], -1).reshape(-1, 16, 16, 4)

    def __call__(self, captions: list[str], seed: int = 0, noise: np.ndarray | None = None, **kw) -> np.ndarray:
        ids = np.array([encode_caption(c, self.vocab) for c in captions], np.int64)
        if noise is None:
            noise = np.random.default_rng(seed).standard_normal((len(captions), 4, 16, 16), dtype=np.float32)
        return self.to_rgba(self.snap(self.sample(ids, noise, **{**self.sampler, **kw})))


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
    ap.add_argument("--scale", type=int, default=8)
    args = ap.parse_args()
    from PIL import Image
    pipe = OnnxPipeline(args.root, args.model)
    unk = [w for c in args.captions for w in words(c) if w not in pipe.vocab]
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
