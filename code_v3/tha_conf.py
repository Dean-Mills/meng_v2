"""
Confidence-aware Type-wise Hungarian Assignment (PEng, 2026-10-29).

predict_tha (eval_hungarian_grouping.py) with one extra cost term. Per joint type, the cost of joint j to cluster k is

    ||e_j - mu_k||  +  lam * |log c_j - log g_k^(-t)|

c_j = the detector's confidence for joint j; g_k^(-t) = geometric-mean confidence of cluster k's joints of the OTHER
types under the previous iteration's labels (leave-type-out, so a joint does not attract itself; the k-means labels
for the first iteration; no term while k has no joint of another type). A log ratio is unchanged when a detector's
confidences are rescaled, so one lam can serve every detector. lam = 0 is predict_tha exactly: same k-means init,
same per-type Hungarian steps, same centroid updates and stopping rule.

Why: the AP-gap anatomy (results.md, 2026-10-29) found that in ~70 % of the slots where the native grouping is right
and THA wrong, the joint native kept is the more confident one; the GAT never sees confidence (node features x, y, v=1).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def _refine(emb_np, k, jt_np, lc, lam, centroids, init_labels, n_iters):
    n = len(emb_np)
    labels = np.zeros(n, dtype=int)
    prev = init_labels
    for iteration in range(n_iters):
        if lam:
            S = np.zeros((k, 17)); N = np.zeros((k, 17))
            np.add.at(S, (prev, jt_np), lc); np.add.at(N, (prev, jt_np), 1)
        for jt in range(17):
            indices = np.where(jt_np == jt)[0]
            if len(indices) == 0:
                continue
            cost = np.linalg.norm(emb_np[indices][:, None] - centroids[None, :], axis=2)
            if lam:
                n_other = N.sum(1) - N[:, jt]
                g = np.where(n_other > 0, (S.sum(1) - S[:, jt]) / np.maximum(n_other, 1), 0.0)
                term = np.abs(lc[indices][:, None] - g[None, :])
                cost = cost + lam * np.where(n_other[None, :] > 0, term, 0.0)
            m = len(indices)
            if m <= k:
                row, col = linear_sum_assignment(cost)
                for r, c in zip(row, col):
                    labels[indices[r]] = c
            else:
                row, col = linear_sum_assignment(cost.T)
                assigned = set()
                for r, c in zip(row, col):
                    labels[indices[c]] = r
                    assigned.add(c)
                for j in range(m):
                    if j not in assigned:
                        labels[indices[j]] = np.argmin(cost[j])
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for i in range(n):
            new_centroids[labels[i]] += emb_np[i]
            counts[labels[i]] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                new_centroids[c] = centroids[c]
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
        prev = labels.copy()
    return labels


def predict_tha_conf(emb_np: np.ndarray, k: int, jt_np: np.ndarray, conf: np.ndarray, lams, n_iters: int = 10):
    """Labels [N] per lam (dict lam -> labels); one k-means initialisation shared by every lam."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(emb_np)
    lc = np.log(np.maximum(np.asarray(conf, np.float64), 1e-6))
    return {lam: _refine(emb_np, k, jt_np, lc, lam, km.cluster_centers_.copy(), km.labels_.astype(int), n_iters)
            for lam in lams}
