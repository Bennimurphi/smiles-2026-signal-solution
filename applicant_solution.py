import json

import numpy as np
from scipy.linalg import svd
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def your_canceller(tx_n, rx):

    # first stage: TX cancellation (ONLY from original rx)
  
    tx_pred = helpers["fit_tx_prediction"](rx)
    resid = rx - tx_pred

    res_band = np.column_stack([
        helpers["score_filter"](resid[:, ch])
        for ch in range(4)
    ])

    N = res_band.shape[0]
    start, end = N // 10, 9 * N // 10
    train = res_band[start:end]

    U, S, Vh = svd(train, full_matrices=False)
    v1 = Vh[0, :]

    shared = res_band @ v1.conj()
    rank1 = np.outer(shared, v1)

    alpha1 = np.clip(
        np.real(np.vdot(res_band[start:end], rank1[start:end]) /
                (np.vdot(rank1[start:end], rank1[start:end]) + 1e-30)),
        0.0, 0.60
    )

    stage1 = rx - tx_pred - alpha1 * rank1


    # Second stage: ONLY spatial refinement (NO re-TX model)
  

    res2_band = np.column_stack([
        helpers["score_filter"](stage1[:, ch])
        for ch in range(4)
    ])

    train2 = res2_band[start:end]

    U2, S2, Vh2 = svd(train2, full_matrices=False)
    v2 = Vh2[0, :]

    shared2 = res2_band @ v2.conj()
    rank2 = np.outer(shared2, v2)

    alpha2 = np.clip(
        np.real(np.vdot(res2_band[start:end], rank2[start:end]) /
                (np.vdot(rank2[start:end], rank2[start:end]) + 1e-30)),
        0.0,
        0.25
    )

    return stage1 - alpha2 * rank2

print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
