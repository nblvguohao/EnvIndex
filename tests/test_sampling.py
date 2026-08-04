"""Tests for distance-stratified environment sampling (improvement #4)."""

from __future__ import annotations

import numpy as np

from envindex.sampling import environment_distance, stratify_sample


def test_environment_distance_clusters():
    # Two directionally-separated clusters (cosine distance picks up the
    # shared direction within a cluster).
    rng = np.random.default_rng(0)
    dir_a = rng.normal(size=16)
    dir_b = rng.normal(size=16)
    z = {f"a{i}": dir_a + rng.normal(scale=0.1, size=16) for i in range(5)}
    z.update({f"b{i}": dir_b + rng.normal(scale=0.1, size=16) for i in range(5)})
    dist = environment_distance(z, k=2)
    within_a = [dist[f"a{i}"] for i in range(5)]
    within_b = [dist[f"b{i}"] for i in range(5)]
    cross_ab = [dist[f"a{i}"] for i in range(5)]  # a's k-NN are within-a
    assert max(within_a + within_b) < 0.5
    assert max(cross_ab) < 0.5


def test_stratify_sample_covers_bins():
    envs = [f"e{i}" for i in range(100)]
    dist = {e: float(i) for i, e in enumerate(envs)}  # uniform spread
    chosen = stratify_sample(envs, dist, n_bins=8, target_total=24, seed=0)
    assert len(set(chosen)) == 24
    # chosen should span the full distance range (low and high ends)
    chosen_dists = sorted(dist[e] for e in chosen)
    assert chosen_dists[0] < 10  # near-min distance included
    assert chosen_dists[-1] > 90  # near-max distance included


def test_stratify_never_exceeds_pool():
    envs = [f"e{i}" for i in range(10)]
    dist = {e: float(i) for i, e in enumerate(envs)}
    chosen = stratify_sample(envs, dist, n_bins=8, target_total=500, seed=0)
    assert len(chosen) <= 10
