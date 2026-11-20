"""Segmentation-perturbation helpers (amendment 2026-11-16). Synthetic arrays only: SYNTHETIC - NOT CLINICAL."""
import numpy as np

import scripts.step0_segmentation_perturbation as sp


def test_perturbations_are_seeded_bounded_and_of_the_stated_kind():
    m = np.zeros((80, 80), bool)
    m[10:70:6, :] = True
    m[:, 40:43] = True
    for cond in sp.CONDITIONS:
        a = sp.perturb(m, cond, np.random.default_rng(3))
        assert a.dtype == bool and a.shape == m.shape
        assert np.array_equal(a, sp.perturb(m, cond, np.random.default_rng(3)))
    d1, d3 = (sp.perturb(m, c, np.random.default_rng(1)) for c in ("dropout1", "dropout3"))
    assert not (d1 & ~m).any() and not (d3 & ~m).any()  # dropout never adds pixels
    assert 0 < (m & ~d1).sum() < (m & ~d3).sum() < 0.08 * m.sum()
    dil = sp.perturb(m, "dilate1", np.random.default_rng(1))
    assert (dil | m).sum() == dil.sum() and dil.sum() > m.sum()
    b = sp.perturb(m, "boundary10", np.random.default_rng(1))
    changed = b ^ m
    grown, shrunk = sp.ndi.binary_dilation(m, sp.CROSS), sp.ndi.binary_erosion(m, sp.CROSS, border_value=1)
    assert changed.any() and not (changed & ~(grown & ~shrunk)).any()  # only boundary pixels flip
    assert (changed & m).any() and (changed & ~m).any()
