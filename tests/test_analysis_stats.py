import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis import stats as S


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 600)
    a = y * 1.0 + rng.normal(0, 1, 600)
    b = y * 0.6 + rng.normal(0, 1, 600)
    return y, a, b


def test_fast_delong_matches_sklearn(data):
    y, a, b = data
    aucs, cov = S.fast_delong(y, np.vstack([a, np.round(b, 1)]))  # include ties
    assert aucs[0] == pytest.approx(roc_auc_score(y, a))
    assert aucs[1] == pytest.approx(roc_auc_score(y, np.round(b, 1)))
    assert cov.shape == (2, 2) and cov[0, 0] > 0


def test_delong_identical_and_different(data):
    y, a, b = data
    assert S.delong_test(y, a, a)[2] == 1.0
    d, z, p = S.delong_test(y, a, b)
    assert d > 0 and z > 0 and p < 0.05


def test_holm():
    np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = rng.uniform(0, 1, 20000) < p
    assert S.ece(y, p) < 0.02
    assert S.ece(y, np.clip(p + 0.3, 0, 1)) > 0.2


def test_platt_recovers_shift():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, 20000)
    y = rng.uniform(size=z.size) < S.sigmoid(z)
    a, b = S.fit_platt(y, z + 1.5)  # model that is shifted by +1.5 logits
    assert a == pytest.approx(1.0, abs=0.08) and b == pytest.approx(-1.5, abs=0.1)


def test_cluster_bootstrap_keeps_clusters_whole():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(10), 3)
    idx = S._resample_idx(rng, 30, cl)
    counts = np.bincount(cl[idx], minlength=10)
    assert (counts % 3 == 0).all()


def test_two_level_diff_sign(data):
    y, a, b = data
    rng = np.random.default_rng(3)
    A = np.vstack([a + rng.normal(0, .05, a.size) for _ in range(3)])
    Bm = np.vstack([b + rng.normal(0, .05, b.size) for _ in range(3)])
    d, lo, hi, p = S.two_level_auc_diff(y, A, Bm, B=200)
    assert d > 0 and lo > 0 and p < 0.05


def test_threshold_for_sens():
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    s = np.arange(12.0)
    t = S.threshold_for_sens(y, s, 0.9)
    assert ((s >= t) & (y == 1)).sum() / y.sum() >= 0.9


def test_threshold_transfer_recovers_shift():
    from analysis.threshold_transfer import hl_shift, loc_scale_thr
    rng = np.random.default_rng(0)
    sp = rng.normal(2.0, 1.0, 400)
    tp = rng.normal(2.0 - 1.5, 1.0, 50)  # target melanomas shifted down by 1.5
    hl, lo = hl_shift(tp, sp)
    assert abs(hl + 1.5) < 0.4 and lo < hl
    # pure shift: loc-scale threshold moves by about the same amount as the median
    t = loc_scale_thr(tp, 0.7, np.median(sp), sp.std())
    assert abs(t - (0.7 - 1.5)) < 0.5


def test_quantile_normalise_invariant_to_monotone_rescaling():
    from analysis.threshold_transfer import quantile_normalise
    x = np.random.default_rng(1).normal(0, 1, 500)
    np.testing.assert_allclose(quantile_normalise(x), quantile_normalise(np.exp(3 * x) - 2))
