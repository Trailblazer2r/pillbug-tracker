"""Tests for the A/B state logic.  Run:  python3 -m unittest test_states"""
import math
import unittest

from pillbug_core import Params, classify, summarise, STATE_A, STATE_A_HID, STATE_B, STATE_B_HID

P = Params(n_sectors=6, cx=0, cy=0, r_in=50, r_out=200, r_ab=130, entry_zone_px=12, min_hidden_s=1.0)
DT = 0.1


def at(r):
    return (float(r), 0.0)


class StateLogic(unittest.TestCase):
    def test_starts_hidden_in_B_until_first_seen(self):
        s = classify([None] * 5 + [at(90)] * 3, DT, P)
        self.assertEqual(set(s[:5]), {STATE_B_HID})
        self.assertEqual(s[5:], [STATE_A] * 3)

    def test_vanishing_deep_in_A_stays_A(self):
        # divider shadow: bug at r=80 (well inside A) disappears for 5 s
        s = classify([at(80)] * 3 + [None] * 50 + [at(82)], DT, P)
        self.assertTrue(all(x in (STATE_A, STATE_A_HID) for x in s))
        self.assertEqual(s[10], STATE_A_HID)

    def test_vanishing_at_boundary_means_shelter(self):
        s = classify([at(90)] * 3 + [at(125)] * 3 + [None] * 30, DT, P)
        self.assertEqual(s[-1], STATE_B_HID)
        self.assertEqual(s[8], STATE_A_HID)   # < 1 s dropout keeps previous side
        self.assertEqual(s[20], STATE_B_HID)  # long dropout near boundary -> shelter

    def test_stays_in_shelter_once_hidden_there(self):
        s = classify([at(125)] + [None] * 40 + [at(100)], DT, P)
        self.assertEqual(s[30], STATE_B_HID)
        self.assertEqual(s[-1], STATE_A)

    def test_visible_in_B(self):
        self.assertEqual(classify([at(160)], DT, P), [STATE_B])

    def test_summary(self):
        obs = [at(90)] * 10 + [None] * 20 + [at(90)] * 10   # 4 s, always A (shadow dropout)
        st = classify(obs, DT, P)
        s = summarise(st, obs, DT, P)
        self.assertAlmostEqual(s["prop_A"], 1.0)
        self.assertEqual(s["n_crossings"], 0)
        self.assertFalse(s["censored"])
        self.assertAlmostEqual(s["time_A_hidden_s"], 2.0, places=3)

    def test_latency_and_censoring(self):
        obs = [None] * 20
        s = summarise(classify(obs, DT, P), obs, DT, P)
        self.assertTrue(s["censored"])
        self.assertEqual(s["prop_A"], 0.0)
        self.assertEqual(s["latency_A_s"], 2.0)

    def test_crossings(self):
        obs = [at(90)] * 5 + [at(125)] * 3 + [None] * 20 + [at(100)] * 5 + [at(125)] * 3 + [None] * 20
        st = classify(obs, DT, P)
        self.assertEqual(summarise(st, obs, DT, P)["n_crossings"], 3)


if __name__ == "__main__":
    unittest.main()
