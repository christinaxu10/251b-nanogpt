# muon.py
# Lightweight single-GPU orthogonalization-based Muon optimizer.
# Adapts Keller Jordan's modded-nanogpt Muon implementation.
# Not DDP-aware. Uses Newton–Schulz backend by default (SVD fallback).

import torch
from torch.optim.optimizer import Optimizer

# ---- zeropower backends ----
def zeropower_via_svd(G):
    # safe, but slow
    U, S, Vt = torch.linalg.svd(G, full_matrices=False)
    return U @ Vt

def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    # Newton-Schulz style orthogonalization (approximate UV^T)
    # Expects a 2D tensor input and returns a same-shape tensor
    a = 3.4445
    b = -4.7750
    c = 2.0315
    X = G.to(torch.bfloat16)
    normval = X.norm() + eps
    X = X / normval
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
    if transposed:
        X = X.t()
    return X.to(G.dtype)

zeropower_backends = {
    "svd": zeropower_via_svd,
    "newtonschulz5": zeropower_via_newtonschulz5
}

# ---- Muon optimizer ----
class Muon(Optimizer):
    """
    Muon: single-GPU orthogonalized update optimizer.
    - Expects 2D parameters for Muon updates (weight matrices)
    - Use standard optimizers (Adam/AdamW) for embeddings / 1D params.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 backend="newtonschulz5", backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            backend = group["backend"]
            backend_steps = group["backend_steps"]
            zeropower = zeropower_backends.get(backend, zeropower_via_newtonschulz5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                state = self.state[p]
                # momentum buffer init
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)
                # momentum update (SGD-momentum style)
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g + momentum * buf
                else:
                    update = buf
                # orthogonalize the update (approximate UV^T)
                # only apply to 2D parameters; if param not 2D, perform normal SGD step
                if p.ndim == 2:
                    try:
                        with torch.cuda.amp.autocast(enabled=False):
                            # run orthogonalization on the update
                            ortho = zeropower(update, steps=backend_steps)
                    except Exception:
                        # fallback to SVD if anything fails
                        ortho = zeropower_via_svd(update)
                    # scale heuristics used in Keller's impl: sqrt(max(1, rows/cols))
                    scale = max(1.0, p.size(0) / max(1.0, p.size(1))) ** 0.5
                    ortho = ortho * scale
                    p.add_(ortho, alpha=-lr)
                else:
                    # 1D or other dims: simple SGD-style step (small lr)
                    p.add_(update, alpha=-lr)
        return loss
