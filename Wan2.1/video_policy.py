"""Optional controls for the 20-step paired Wan benchmark."""

import torch
import torch.nn.functional as F
from budget_policy import WindowBudget


class VideoPolicy:
    def __init__(self, mode, profile=None, probe=None, replay=None):
        self.mode, self.profile, self.probe, self.replay = mode, profile, probe, replay
        self.trace = []
        self.scores = []
        self.budget = WindowBudget()
        self.history = {True: [], False: []}

    def scores_at(self, model, z, t):
        if model.cached_z is None:
            return (0.0, 0.0)
        delta = (z - model.cached_z).float()
        time_part = model.cached_J_t_norm * (t - model.cached_t).abs().item()
        global_part = model.cached_J_z_norm * delta.norm().item()
        # Same units as the SenCache global L2 score. Pool over spatial patches at each latent frame.
        field = delta.square().mean(0, keepdim=True).unsqueeze(0)
        kernel = (1, min(8, field.shape[-2]), min(8, field.shape[-1]))
        blocks = F.avg_pool3d(field, kernel, ceil_mode=True, count_include_pad=False).flatten()
        top_rms = blocks.topk(max(1, blocks.numel() // 8)).values.mean().sqrt().item()
        local_part = model.cached_J_z_norm * top_rms * delta.numel() ** 0.5
        return global_part + time_part, 0.5 * (global_part + local_part) + time_part

    def skip(self, model, step, z, t):
        if self.mode == "calibrate":
            self.scores.append(self.scores_at(model, z, t))
            return False
        if self.mode == "probe":
            return step == self.probe
        if self.replay is not None:
            return self.replay[step] == "reuse"
        if self.mode == "fixed":
            return step in (2, 5, 8, 11, 14, 17)
        if not self.budget.eligible(step):
            return False
        g, a = self.scores_at(model, z, t)
        return self.budget.choose(step, self.mode, g, a, self.profile)

    def remember(self, cond, step, residual):
        if self.mode in ("C", "C_guard"):
            hist = self.history[cond]
            hist.append((step, residual.detach()))
            del hist[:-2]

    def estimate(self, cond, step, last):
        hist = self.history[cond]
        if len(hist) < 2:
            return last
        (older, old_r), (recent, new_r) = hist
        ratio = (step - recent) / (recent - older)
        direction = (new_r.float() - old_r.float()) * ratio
        if self.mode == "C_guard" and (
            recent - older > 2 or direction.norm() > 0.15 * last.float().norm()
        ):
            return last
        return (last.float() + direction).to(last.dtype)
