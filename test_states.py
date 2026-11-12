"""Tests for the A/B state logic and the tracker.  Run:  python3 -m unittest test_states"""
import unittest

import cv2
import numpy as np

from pillbug_core import (Params, classify, summarise, detect, sector_cut, ChamberTracker,
                          build_sector_map, assign_candidates,
                          STATE_A, STATE_A_HID, STATE_B, STATE_B_HID, IN_A)

P = Params(n_sectors=6, cx=0, cy=0, r_in=50, r_out=200, r_ab=130, entry_zone_px=12,
           min_hidden_s=1.0, min_dwell_s=2.0, hyst_px=12)
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
        self.assertTrue(all(x in IN_A for x in s))
        self.assertEqual(s[10], STATE_A_HID)

    def test_vanishing_at_boundary_means_shelter(self):
        s = classify([at(90)] * 3 + [at(125)] * 3 + [None] * 40, DT, P)
        self.assertEqual(s[8], STATE_A_HID)   # < 1 s dropout keeps previous side
        self.assertEqual(s[30], STATE_B_HID)  # long dropout near boundary -> shelter (dated back)
        self.assertEqual(s[-1], STATE_B_HID)

    def test_stays_in_shelter_then_comes_back(self):
        s = classify([at(125)] + [None] * 40 + [at(100)] * 30, DT, P)
        self.assertEqual(s[30], STATE_B_HID)
        self.assertEqual(s[-1], STATE_A)
        self.assertEqual(s[42], STATE_A)       # the return is dated back to its first frame

    def test_long_absence_deep_in_A_eventually_means_B(self):
        p = Params(n_sectors=6, cx=0, cy=0, r_in=50, r_out=200, r_ab=130, entry_zone_px=12, max_shadow_s=60)
        s = classify([at(80)] * 5 + [None] * 1500, DT, p)       # 150 s out of sight
        self.assertEqual(s[300], STATE_A_HID)                    # 30 s: still shadow
        self.assertEqual(s[550], STATE_A_HID)                    # just under 60 s
        self.assertEqual(s[900], STATE_B_HID)
        self.assertEqual(s[-1], STATE_B_HID)

    def test_visible_in_B(self):
        self.assertEqual(classify([at(160)], DT, P), [STATE_B])

    def test_short_blip_does_not_flip(self):
        s = classify([at(100)] * 30 + [at(160)] * 5 + [at(100)] * 30, DT, P)
        self.assertTrue(all(x in IN_A for x in s))

    def test_flicker_at_the_edge_is_one_state(self):
        # visible near the boundary, vanishing for 1.2 s, again and again: must not toggle
        obs = [at(100)] * 20 + ([at(128)] * 5 + [None] * 12) * 12
        s = classify(obs, DT, P)
        self.assertTrue(all(x in IN_A for x in s))
        self.assertEqual(summarise(s, obs, DT, P)["n_crossings"], 0)

    def test_hysteresis_band(self):
        # hovering around the boundary (visible) never switches
        s = classify([at(100)] * 10 + [at(136), at(124)] * 30, DT, P)
        self.assertTrue(all(x in IN_A for x in s))

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
        obs = ([at(90)] * 40 + [at(125)] * 5 + [None] * 40 + [at(100)] * 40 + [at(125)] * 5 + [None] * 40)
        st = classify(obs, DT, P)
        self.assertEqual(summarise(st, obs, DT, P)["n_crossings"], 3)


class Tracker(unittest.TestCase):
    def setUp(self):
        self.p = Params(cx=0, cy=0, r_in=50, r_out=200, cup_diam_cm=5)   # 20 px / cm
        self.t = ChamberTracker(DT, self.p)

    def cand(self, x, y):
        return (float(x), float(y), 400, 80.0)

    def lock(self, x, y):
        for _ in range(self.t.confirm_n):
            self.t.update([self.cand(x, y)])

    def test_follows_nearest(self):
        self.lock(100, 100)
        o = self.t.update([self.cand(103, 100), self.cand(150, 100)])
        self.assertEqual(o, (103.0, 100.0))

    def test_flash_far_away_is_ignored(self):
        self.lock(100, 100)
        for _ in range(4):
            self.assertIsNone(self.t.update([self.cand(100, 200)]))      # flash, gone before confirm
        self.assertEqual(self.t.update([self.cand(101, 100)]), (101.0, 100.0))

    def test_far_detection_is_accepted_when_it_persists(self):
        self.lock(100, 100)
        out = [self.t.update([self.cand(100, 200)]) for _ in range(12)]
        self.assertIsNone(out[0])
        self.assertEqual(out[self.t.confirm_n - 1], (100.0, 200.0))

    def test_first_sighting_is_confirmed_then_backfilled(self):
        out = [self.t.update([self.cand(100, 100)]) for _ in range(self.t.confirm_n)]
        self.assertTrue(all(o is None for o in out[:-1]))
        self.assertEqual(out[-1], (100.0, 100.0))
        self.assertEqual(len(self.t.backfill), self.t.confirm_n - 1)


class DividerBorder(unittest.TestCase):
    def setUp(self):
        self.p = Params(n_sectors=4, cx=200, cy=200, r_in=30, r_out=190, cup_diam_cm=5)
        self.spokes = [0.0, 90.0, 180.0, 270.0]          # sector k spans spokes[k] .. spokes[k+1]
        self.sec = build_sector_map((400, 400), self.p, self.spokes)
        self.trk = [ChamberTracker(DT, self.p) for _ in range(4)]

    def at_angle(self, deg, r=100.0):
        a = np.radians(deg)
        return (200 + r * np.cos(a), 200 + r * np.sin(a), 500, 80.0)

    def test_plain_blob_goes_to_its_sector(self):
        by = assign_candidates([self.at_angle(45)], self.sec, self.trk, self.p, self.spokes)
        self.assertEqual([len(b) for b in by], [1, 0, 0, 0])

    def test_blob_on_divider_goes_to_the_chamber_that_follows_it(self):
        self.trk[3].last = self.at_angle(357)[:2]          # chamber 4's bug is just across the divider
        by = assign_candidates([self.at_angle(3)], self.sec, self.trk, self.p, self.spokes)
        self.assertEqual([len(b) for b in by], [0, 0, 0, 1])

    def test_blob_on_divider_without_a_follower_stays_by_position(self):
        by = assign_candidates([self.at_angle(3)], self.sec, self.trk, self.p, self.spokes)
        self.assertEqual([len(b) for b in by], [1, 0, 0, 0])

    def test_nearer_follower_wins(self):
        self.trk[0].last = self.at_angle(4, 100)[:2]       # own chamber's bug is even closer
        self.trk[3].last = self.at_angle(357, 100)[:2]
        by = assign_candidates([self.at_angle(3)], self.sec, self.trk, self.p, self.spokes)
        self.assertEqual([len(b) for b in by], [1, 0, 0, 0])

    def test_a_tracker_that_lost_its_bug_cannot_claim(self):
        self.trk[3].last = self.at_angle(357)[:2]
        self.trk[3].gap = 40                                   # 4 s without seeing it
        by = assign_candidates([self.at_angle(3)], self.sec, self.trk, self.p, self.spokes)
        self.assertEqual([len(b) for b in by], [1, 0, 0, 0])

    def test_far_from_divider_is_never_moved(self):
        self.trk[3].last = self.at_angle(300)[:2]
        by = assign_candidates([self.at_angle(30)], self.sec, self.trk, self.p, self.spokes)
        self.assertEqual([len(b) for b in by], [1, 0, 0, 0])


class Detection(unittest.TestCase):
    def scene(self, centres, r=8):
        bg = np.full((200, 200), 150, np.uint8)
        img = bg.copy()
        for c in centres:
            cv2.circle(img, c, r, 60, -1)
        return img, bg

    def test_merged_bugs_are_split(self):
        roi = np.ones((200, 200), bool)
        p = Params(min_area=50, max_area=2000)
        one = detect(*self.scene([(60, 100)]), roi, p)
        self.assertEqual(len(one), 1)
        p.bug_area = one[0][2]
        two = detect(*self.scene([(100, 100), (117, 100)]), roi, p)
        self.assertEqual(len(two), 2)
        self.assertTrue(abs(two[0][0] - two[1][0]) > 10)

    def test_sector_border_separates_touching_bugs(self):
        roi = np.ones((200, 200), bool)
        sec = np.zeros((200, 200), np.int8)
        sec[:, 100:] = 1
        d = detect(*self.scene([(93, 100), (107, 100)]), roi, Params(min_area=30, max_area=2000), sector_cut(sec))
        self.assertEqual(len(d), 2)
        self.assertTrue(min(x for x, *_ in d) < 100 < max(x for x, *_ in d))

    def test_thin_shadow_line_is_not_a_bug(self):
        bg = np.full((200, 200), 150, np.uint8)
        img = bg.copy()
        cv2.line(img, (20, 100), (180, 104), 60, 6)
        d = detect(img, bg, np.ones((200, 200), bool), Params(min_area=50, max_area=5000))
        self.assertEqual(d, [])

    def test_exposure_drift_does_not_light_up_dark_areas(self):
        bg = np.full((200, 200), 150, np.uint8)
        bg[:, :20] = 25                       # a dark divider / roof shadow
        img = (bg.astype(np.float32) * 1.2).astype(np.uint8)   # camera got brighter
        self.assertEqual(detect(img, bg, np.ones((200, 200), bool), Params(min_area=50)), [])


if __name__ == "__main__":
    unittest.main()
