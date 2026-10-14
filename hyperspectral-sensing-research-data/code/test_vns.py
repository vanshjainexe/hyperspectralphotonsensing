"""Fast physics/accounting and software invariant checks."""

import unittest
import numpy as np
from vns_robustness import (
    operator,
    mixing,
    quantize,
    sense,
    Config,
    PAIRS,
    seed,
    fit,
    predict,
    measure,
)


class TestForwardModel(unittest.TestCase):
    def test_photon_conservation(self):
        for k in [3, 4, 6, 8, 12, 16, 24]:
            for fractional in [False, True]:
                p = operator(k, fractional)
                np.testing.assert_allclose(p.sum(0), np.ones(31) / 31)
                self.assertTrue(np.all(p >= 0))
                self.assertAlmostEqual(float(p.sum()), 1.0)

    def test_crosstalk_conservation(self):
        for k in [3, 6, 24]:
            m = mixing(k, 0.15)
            np.testing.assert_allclose(m.sum(0), 1.0)
            self.assertTrue(np.all(m >= 0))

    def test_fixed_bit_budgets(self):
        for budget, pairs in PAIRS.items():
            for k, b in pairs:
                self.assertEqual(k * b, budget)

    def test_white_reference_ratio(self):
        for k in [3, 4, 24]:
            p = operator(k)
            x = np.ones((4, 31))
            c = Config("test")
            r = sense(
                x, p, 1000, k, 8, "router", c, np.random.default_rng(0), False, False
            )
            f = sense(
                x, p, 1000, k, 8, "filter", c, np.random.default_rng(0), False, False
            )
            np.testing.assert_allclose(r[0], 1.0)
            np.testing.assert_allclose(f[0], 1.0)
            self.assertAlmostEqual(r[3] / f[3], 0.84 * k)

    def test_quantization_codes_and_bounds(self):
        for bits in [1, 2, 8]:
            for kind in ["uniform", "sqrt"]:
                counts = np.linspace(-10, 200, 100000)
                q = quantize(counts, 100, bits, kind)
                self.assertLessEqual(len(np.unique(q)), 2**bits)
                self.assertTrue(np.all(np.diff(q) >= 0))
                self.assertEqual(q[0], 0.0)
                self.assertEqual(q[-1], 100.0)

    def test_poisson_mean(self):
        x = np.full((100000, 31), 0.4)
        p = operator(4)
        y, _, _, e = sense(
            x,
            p,
            100,
            4,
            8,
            "router",
            Config("x"),
            np.random.default_rng(18),
            True,
            False,
        )
        self.assertLess(abs(y.mean() - 0.4), 0.002)
        self.assertAlmostEqual(e, 33.6)

    def test_repeatability(self):
        x = np.full((100, 31), 0.5)
        p = operator(6)
        c = Config("test")
        a = sense(x, p, 100, 6, 4, "router", c, np.random.default_rng(seed(1, 2)))
        b = sense(x, p, 100, 6, 4, "router", c, np.random.default_rng(seed(1, 2)))
        np.testing.assert_array_equal(a[0], b[0])

    def test_noiseless_identity(self):
        x = np.random.default_rng(3).random((500, 31))
        d = fit(x, x, 1e-8)
        z = predict(x, d)
        self.assertLess(measure(x, z)["rmse"], 1e-8)

    def test_sam_zero_prediction_not_omitted(self):
        m = measure(np.ones((4, 31)), np.zeros((4, 31)))
        self.assertEqual(m["sam"], 90.0)
        self.assertEqual(m["sam_pixels"], 4)

    def test_darkness(self):
        p = operator(6)
        x = np.zeros((100, 31))
        y, _, _, e = sense(
            x, p, 100, 6, 4, "router", Config("x"), np.random.default_rng(2)
        )
        self.assertEqual(float(y.max()), 0.0)
        self.assertEqual(e, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
