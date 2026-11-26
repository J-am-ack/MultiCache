"""CFG output-gap reuse for the serial conditional/unconditional Wan sampler."""
import torch


class CFGCache:
    def __init__(self, scale, tolerance, max_age):
        self.scale = scale
        self.tolerance = tolerance
        self.max_age = max_age
        self.reset()

    def reset(self):
        self.delta = None
        self.drift = float("inf")
        self.anchor = -1
        self.conditional = None
        self.last_refresh_single = False
        self.counts = dict(reuse=0, single=0, full=0)

    def choose(self, step, skip, last_step):
        if skip:
            return "reuse"
        if (step < last_step and self.delta is not None
                and step - self.anchor <= self.max_age
                and self.drift <= self.tolerance):
            return "single"
        return "full"

    def observe_pair(self, step, cond, uncond):
        delta = (cond - uncond).detach()
        if self.delta is not None:
            # Historical proxy for relative guided-output error, measurable on full pairs.
            guided = cond + (self.scale - 1) * delta
            self.drift = ((self.scale - 1) * torch.linalg.vector_norm(delta - self.delta)
                          / torch.linalg.vector_norm(guided).clamp_min(1e-8)).item()
        self.delta = delta
        self.anchor = step
        self.last_refresh_single = False

    def estimate_unconditional(self, cond):
        return cond - self.delta

