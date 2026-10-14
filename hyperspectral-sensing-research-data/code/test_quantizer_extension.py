import unittest
import numpy as np
from vns_quantizer_extension import fit_quantile_adc, apply_adc
from vns_robustness import operator, sense, Config


class CalibrationChecks(unittest.TestCase):
    def test_zero_atom_preserves_photon_detection(self):
        y = np.r_[np.zeros(90), np.ones(10)][:, None]
        q = fit_quantile_adc(y, 1)
        out, c = apply_adc(y, q)
        self.assertEqual(len(np.unique(c)), 2)
        np.testing.assert_array_equal(out, y)

    def test_budget_and_monotonicity(self):
        y = np.sort(np.random.default_rng(2).gamma(0.5, 0.4, (1200, 3)), axis=0)
        for b in [1, 2, 4, 8]:
            q = fit_quantile_adc(y, b)
            out, c = apply_adc(y, q)
            self.assertTrue(np.isfinite(out).all())
            self.assertTrue((np.diff(out, axis=0) >= -1e-12).all())
            self.assertTrue(all(len(np.unique(col)) <= 2**b for col in c.T))

    def test_constant_training_distribution(self):
        y = np.zeros((100, 4))
        out, _ = apply_adc(y, fit_quantile_adc(y, 1))
        np.testing.assert_array_equal(out, y)

    def test_matched_detected_photons(self):
        x = np.random.default_rng(11).random((500, 31))
        cfg = Config("main")
        for k in [3, 4, 6, 8, 12, 24]:
            p = operator(k)
            n = 1000
            b = 24 // k
            router = sense(x, p, n, k, b, "router", cfg, np.random.default_rng(5))[0]
            filt = sense(
                x, p, 0.84 * k * n, k, b, "filter", cfg, np.random.default_rng(5)
            )[0]
            np.testing.assert_allclose(router, filt, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
