"""Three-step refresh budgets and calibration shared by the image and video runs."""

import numpy as np


def fit_profiles(observations):
    """Each entry contains dense step scores and single-skip final-output MSE."""
    damage = np.mean([case["mse"] for case in observations], axis=0)
    importance = np.sqrt(np.maximum(damage, 1e-6) / max(np.median(damage[1:19]), 1e-6))
    scores = np.array([case["scores"] for case in observations])
    values = {
        "global": scores[:, :, 0],
        "local": scores[:, :, 1],
        "weighted": scores[:, :, 0] * importance,
    }
    cutoffs = {
        key: [float(np.quantile(arr[:, 1 + 3 * w : 4 + 3 * w], 1 / 3)) for w in range(6)]
        for key, arr in values.items()
    }
    return dict(importance=importance.tolist(), cutoffs=cutoffs, observations=observations)


class WindowBudget:
    """Spend one skip in each group of three candidates, without future latents."""

    def __init__(self):
        self.used_window = -1

    def eligible(self, step):
        return 1 <= step <= 18 and self.used_window != (step - 1) // 3

    def choose(self, step, mode, global_score, local_score, profile):
        window = (step - 1) // 3
        key = {"global": "global", "A": "local", "B": "weighted"}[mode]
        value = {
            "global": global_score,
            "local": local_score,
            "weighted": global_score * profile["importance"][step],
        }[key]
        if value <= profile["cutoffs"][key][window] or step % 3 == 0:
            self.used_window = window
            return True
        return False
