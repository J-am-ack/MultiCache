"""Regression checks for compute accounting and the two guarded estimates."""

from types import SimpleNamespace
import torch
from bench import Policy


class ToyDenoiser:
    def __init__(self):
        self.calls = []

    def forward(self, x, t, encoder_hidden_states, added_cond_kwargs, **kwargs):
        self.calls.append(len(x))
        return (x + encoder_hidden_states,)


def main():
    net = ToyDenoiser()
    pipe = SimpleNamespace(unet=net)
    profile = dict(
        importance=[1.0] * 20, cutoffs={key: [0.1] * 6 for key in ("global", "local", "weighted")}
    )
    ctx = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1)
    for mode in ("fixed", "global", "A", "B", "C", "C_guard", "D", "serial20"):
        policy = Policy(mode, profile)
        net.calls.clear()
        with policy.attach(pipe):
            for i in range(20):
                x = torch.full((2, 1, 16, 16), 100 + i, dtype=torch.float32)
                u, c = net.forward(x, i, encoder_hidden_states=ctx, added_cond_kwargs={})[0]
                assert (c - u).mean() == 1
                if mode == "C" and policy.trace[-1] == "predict":
                    assert torch.allclose(u, x[0])
        actual = sum(net.calls)
        reported = policy.trace.count("full") * 2 + policy.trace.count("single")
        assert actual == reported
        if mode == "D":
            assert 14 < actual < 28
        else:
            expected = 40 if mode == "serial20" else 28
            assert actual == expected
        if mode in ("A", "B", "global"):
            for first in (1, 4, 7, 10, 13, 16):
                assert policy.trace[first : first + 3].count("reuse") == 1
    guarded = Policy("C_guard")
    guarded.previous, guarded.last = torch.zeros(2), torch.ones(2)
    guarded.previous_i, guarded.last_i = 0, 1
    assert guarded.reuse(2)[1] == "fallback"
    print("Verified branch counts, window budgets, CFG branch order, linear prediction and guard.")


if __name__ == "__main__":
    main()
