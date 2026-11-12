"""Core analysis for the pill bug light/shelter arena.

Design (see pillbug_experiment_summary.md):
  * circular arena split into N radial sectors ("chambers"), one focal bug each
  * centre cup (not scored), region A = lit open area next to the cup,
    region B = roofed shelter on the outside. From above only A is visible; the
    black roof's inner edge is the A/B boundary (equal-area design: r_AB=sqrt((R^2+a^2)/2)).
  * the bug seen in a chamber is ALWAYS that chamber's bug (identity = chamber)

The roofed shelter hides the bug from the overhead camera, so "not visible" has
to be interpreted. A bug that vanishes deep inside A (e.g. in the shadow of a
radial divider) is still counted as being in A; only a bug that was last seen
at/near the A/B boundary (or inside B) and then vanishes counts as in B.
"""
from dataclasses import dataclass, field, replace
import csv
import math
import os

import cv2
import numpy as np

STATE_A, STATE_A_HID, STATE_B, STATE_B_HID = "A", "A_hidden", "B", "B_hidden"
IN_A = (STATE_A, STATE_A_HID)


@dataclass
class Params:
    n_sectors: int = 8
    cx: float = 666.0
    cy: float = 369.0
    r_in: float = 84.0          # edge of the centre cup (px)
    r_out: float = 240.0        # inner wall of the arena (px)
    r_ab: float = None          # A/B boundary (px); None -> inner edge of the roof (= r_out)
    number_shift: int = 0       # rotate chamber numbering
    step: int = 3               # analyse every Nth frame
    thresh: int = 18            # background-difference threshold (gray levels)
    min_area: int = 120
    max_area: int = 2500
    entry_zone_px: float = 45.0   # last seen this close to the roof edge (or beyond) -> entered B. Pilot: long
                                  # disappearances start at r<120 (stationary at the cup) or r>195 (roof), none between
    min_hidden_s: float = 1.0     # shorter dropouts keep the previous state
    min_dwell_s: float = 2.0      # an A<->B switch is only accepted after the new side has held this long
                                  # (the switch is dated back to when it started)
    max_shadow_s: float = 120.0   # not seen while last seen deep in A: still A for this long (divider shadow), then B
                                  # (the dark bands along the dividers lead to the roof, so a long absence means it left)
    hyst_px: float = 12.0         # visible within this distance of the A/B boundary counts as "no change"
    min_contrast: int = 30        # a blob must have a core at least this far from the background
    bug_area: float = None        # area of ONE bug in px (None -> measured from the video); bigger blobs are split
    border_px: float = 35.0       # a blob this close to a divider goes to the chamber that was following it
    confirm_s: float = 1.0        # a detection that jumps far from the last position is trusted only after this long
    max_speed_cm_s: float = 3.0   # fastest plausible bug speed, sets how far a bug may move between frames
    cup_diam_cm: float = 5.0      # centre cup diameter, sets the px -> cm scale
    start_s: float = 0.0
    max_minutes: float = None
    empty_bg: str = ""            # optional image/video of the EMPTY arena
    empty_first_s: float = 0.0    # or: first N seconds of the video are empty
    save_video: bool = True

    def boundary(self):
        # The black roof covers region B. Everything visible from above is A, so the
        # A/B boundary is the inner edge of the roof unless set explicitly.
        return float(self.r_ab) if self.r_ab else float(self.r_out)

    def px_per_cm(self):
        return 2.0 * self.r_in / self.cup_diam_cm


# ----------------------------------------------------------------- geometry
def auto_geometry(bg):
    """Best-effort detection of the centre cup and the roof's inner edge.
    Returns dict(cx, cy, r_in, r_out) or None. The GUI lets the user drag to fine-tune."""
    h, w = bg.shape
    g = cv2.medianBlur(bg, 5)
    s = h / 720.0

    def hough(r0, r1):
        cs = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=80, param2=40,
                              minRadius=int(r0 * s), maxRadius=int(r1 * s))
        return [] if cs is None else cs[0][:6].tolist()

    roofs, cups = hough(200, 280), hough(60, 110)
    if not roofs or not cups:
        return None
    best = min(((math.hypot(r[0] - c[0], r[1] - c[1]), r, c) for r in roofs for c in cups), key=lambda t: t[0])
    if best[0] > 40 * s:
        return None
    _, r, c = best
    return dict(cx=(r[0] + c[0]) / 2, cy=(r[1] + c[1]) / 2, r_in=float(c[2]), r_out=float(r[2]))


def guess_n_sectors(bg, p, candidates=(12, 10, 8, 6, 5, 4), min_tooth=0.35):
    """Largest divider count for which EVERY tooth of the comb sits on a dark divider
    (fewer teeth would trivially match a subset of the real dividers)."""
    r, ang = _polar(bg.shape, p.cx, p.cy)
    ring = (r > p.r_in + 10) & (r < p.boundary() - 10)
    dark = ring & (bg < np.median(bg[ring]) - 25)
    if dark.sum() < 200:
        return p.n_sectors
    hist, _ = np.histogram(ang[dark], bins=360, range=(0, 360))
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, "same")[3:-3].astype(float)
    for n in candidates:
        step = 360.0 / n
        weakest = max(min(hist[int(round(phi + k * step)) % 360] for k in range(n))
                      for phi in range(int(math.ceil(step))))
        if weakest >= min_tooth * hist.max():
            return n
    return p.n_sectors


def _polar(shape, cx, cy):
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return np.hypot(xx - cx, yy - cy), (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360) % 360


def find_spoke_angles(bg, p):
    """Angles (deg, image coords, clockwise from 3 o'clock) of the N dark dividers."""
    n = p.n_sectors
    step = 360.0 / n
    r, ang = _polar(bg.shape, p.cx, p.cy)
    ring = (r > p.r_in + 10) & (r < p.boundary() - 10)  # region A only: the B roof is dark everywhere
    level = np.median(bg[ring])
    dark = ring & (bg < level - 25)
    if dark.sum() < 200:
        return [(k * step) % 360 for k in range(n)]
    hist, _ = np.histogram(ang[dark], bins=360, range=(0, 360))
    hist = np.convolve(np.r_[hist[-3:], hist, hist[:3]], np.ones(5) / 5, "same")[3:-3]
    comb = [sum(hist[int(round(phi + k * step)) % 360] for k in range(n)) for phi in range(int(math.ceil(step)))]
    phi0 = int(np.argmax(comb))
    out = []
    for k in range(n):
        nominal = phi0 + k * step
        win = [int(round(nominal + d)) % 360 for d in range(-int(step // 4), int(step // 4) + 1)]
        out.append(float(win[int(np.argmax(hist[win]))]) if hist[win].max() > 0.2 * hist.max() else nominal % 360)
    return sorted(a % 360 for a in out)


def build_sector_map(shape, p, spokes):
    """int8 map: sector index 0..N-1 inside the scored annulus, -1 elsewhere."""
    r, ang = _polar(shape, p.cx, p.cy)
    sec = np.full(shape, -1, np.int8)
    n = p.n_sectors
    for k in range(n):
        a0, a1 = spokes[k], spokes[(k + 1) % n]
        if a1 <= a0:
            a1 += 360
        a = np.where(ang < a0, ang + 360, ang)
        sec[(a >= a0) & (a < a1)] = k
    sec[~((r > p.r_in + 3) & (r < p.r_out - 3))] = -1
    return sec


def chamber_label(k, p):
    return (k + p.number_shift) % p.n_sectors + 1


def draw_overlay(img, p, spokes, sec=None, labels=True):
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if sec is not None:
        tint = np.zeros_like(out)
        tint[sec >= 0] = (0, 140, 0)
        # B = shelter under the roof (only drawn for orientation; it is NOT tracked)
        rr, _ = _polar(out.shape[:2], p.cx, p.cy)
        tint[(rr >= p.boundary()) & (rr < p.boundary() * 1.37)] = (160, 60, 0)
        out = cv2.addWeighted(out, 1, tint, 0.28, 0)
    c = (int(round(p.cx)), int(round(p.cy)))
    cv2.circle(out, c, int(p.r_in), (0, 0, 255), 1)
    cv2.circle(out, c, int(p.r_out), (0, 0, 255), 1)
    cv2.circle(out, c, int(round(p.boundary())), (0, 200, 255), 2)  # A/B boundary
    for a in spokes:
        t = math.radians(a)
        cv2.line(out, (int(p.cx + p.r_in * math.cos(t)), int(p.cy + p.r_in * math.sin(t))),
                 (int(p.cx + p.r_out * math.cos(t)), int(p.cy + p.r_out * math.sin(t))), (255, 160, 0), 1)
    if labels:
        for k in range(p.n_sectors):
            a0 = spokes[k]
            span = (spokes[(k + 1) % p.n_sectors] - a0) % 360
            t = math.radians(a0 + span / 2)
            rr = (p.r_in + p.boundary()) / 2
            pos = (int(p.cx + rr * math.cos(t)) - 8, int(p.cy + rr * math.sin(t)) + 8)
            cv2.putText(out, str(chamber_label(k, p)), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return out


# --------------------------------------------------------------- background
def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def median_background(cap, f0, f1, n_samples=40):
    idxs = np.linspace(f0, max(f0, f1 - 1), n_samples).astype(int)
    stack = []
    for i in dict.fromkeys(idxs.tolist()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            stack.append(_gray(f))
    if not stack:
        raise RuntimeError("could not read frames for the background")
    return np.median(stack, 0).astype(np.uint8)


def load_empty_background(path, shape):
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise RuntimeError(f"cannot read {path}")
    else:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        g = median_background(cap, 0, n)
        cap.release()
    if g.shape != shape:
        g = cv2.resize(g, (shape[1], shape[0]))
    return g


def quick_background(video, n=15):
    cap = cv2.VideoCapture(video)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = median_background(cap, 0, nf, n)
    first = None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first = cap.read()
    cap.release()
    return bg, first


# ---------------------------------------------------------------- detection
def sector_cut(sec):
    """1-px lines along the sector borders, so that one blob can never span two chambers."""
    e = np.zeros(sec.shape, bool)
    e[:, 1:] |= sec[:, 1:] != sec[:, :-1]
    e[1:, :] |= sec[1:, :] != sec[:-1, :]
    return cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def detect(gray, bg, roi, p, cut=None):
    """Blobs that differ from the background -> list of (x, y, area, strength).

    * exposure drift is removed with a gain (not an offset), so dark places (dividers, roof
      shadow) do not light up when the camera brightness changes
    * `cut` (optional sector borders) separates touching bugs of neighbouring chambers; the analysis
      does not use it, because it also cuts one bug lying on a divider into two halves
    * blobs without a strong core, or long thin ones (shadow edges), are not bugs
    * a blob much larger than one bug (`p.bug_area`) is several bugs: it is split with k-means
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    b = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.float32)
    gain = float(np.median(g[roi])) / max(1.0, float(np.median(b[roi])))
    diff = g - gain * b
    diff -= float(np.median(diff[roi]))
    ad = np.abs(diff)
    m = ((ad > p.thresh) & roi).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    if cut is not None:
        m[cut] = 0
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m)
    bug = p.bug_area
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < p.min_area or area > 3 * p.max_area:
            continue
        x0, y0, w0, h0 = (int(v) for v in stats[i, :4])
        ys, xs = np.nonzero(lab[y0:y0 + h0, x0:x0 + w0] == i)
        ys, xs = ys + y0, xs + x0
        strength = float(np.percentile(ad[ys, xs], 90))
        if strength < p.min_contrast:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if bug and area > 1.55 * bug:
            k = int(min(4, max(2, round(area / bug))))
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, lbl, ctr = cv2.kmeans(pts, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
            lbl = lbl.ravel()
            for j in range(k):
                cnt = int((lbl == j).sum())
                if cnt >= 0.5 * p.min_area:
                    out.append((float(ctr[j][0]), float(ctr[j][1]), cnt, strength))
            continue
        if area > p.max_area:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        if min(rw, rh) > 0 and max(rw, rh) / min(rw, rh) > 4.5:
            continue
        out.append((float(cents[i][0]), float(cents[i][1]), area, strength))
    return out


def estimate_bug_area(cap, f0, f1, bg, roi, cut, p, n=60):
    """Typical blob area = one bug. Measured without splitting, so it must be called with bug_area=None."""
    areas = []
    for i in dict.fromkeys(np.linspace(f0, max(f0, f1 - 1), n).astype(int).tolist()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            areas += [d[2] for d in detect(_gray(f), bg, roi, p, cut)]
    return float(np.median(areas)) if len(areas) >= 10 else None


class ChamberTracker:
    """Chooses the focal bug among one chamber's candidates, with continuity.

    A bug cannot jump: a candidate farther from the last position than the bug could have
    walked is only believed after it has stayed put for `confirm_s` (this removes flashes,
    reflections and the like, and stops the track from hopping onto an artefact).
    """

    def __init__(self, dt, p):
        self.dt = dt
        self.v_px = p.max_speed_cm_s * p.px_per_cm()
        self.confirm_n = max(1, int(round(p.confirm_s / dt)))
        self.same_px = 0.5 * p.px_per_cm()
        self.max_jump = 2.0 * p.r_out
        self.last, self.gap = None, 0
        self.pending, self.hist = None, []
        self.backfill = []      # after a confirmation: positions of the unconfirmed frames just before this one

    def claim_dist(self, c):
        """Distance to where this chamber's bug was, if c could plausibly be it (else None)."""
        if self.last is None or self.gap * self.dt > 1.0:     # only a bug that is being followed right now
            return None
        d = math.hypot(c[0] - self.last[0], c[1] - self.last[1])
        return d if d <= 25.0 + 0.5 * self.v_px * self.gap * self.dt else None

    def _accept(self, c):
        self.last, self.gap = (c[0], c[1]), 0
        self.backfill = self.hist[:-1] if self.hist else []
        self.pending, self.hist = None, []
        return (c[0], c[1])

    def update(self, cands):
        self.backfill = []
        if not cands:
            self.gap += 1
            self.pending, self.hist = None, []
            return None
        if self.last is not None:
            allowed = min(self.max_jump, 25.0 + self.v_px * (self.gap + 1) * self.dt)
            near = [c for c in cands if math.hypot(c[0] - self.last[0], c[1] - self.last[1]) <= allowed]
            if near:
                return self._accept(min(near, key=lambda c: math.hypot(c[0] - self.last[0], c[1] - self.last[1])))
        c = max(cands, key=lambda c: c[2] * c[3])
        if self.pending is None or math.hypot(c[0] - self.pending[0], c[1] - self.pending[1]) > self.same_px:
            self.hist = []
        self.pending = (c[0], c[1])
        self.hist.append(self.pending)
        if len(self.hist) >= self.confirm_n:
            return self._accept(c)
        self.gap += 1
        return None


def border_info(x, y, p, spokes, k):
    """Chamber on the other side of the divider nearest to (x, y), and the distance (px) to that divider."""
    n = len(spokes)
    r = math.hypot(x - p.cx, y - p.cy)
    a = math.degrees(math.atan2(y - p.cy, x - p.cx)) % 360
    best = None
    for j, s in enumerate(spokes):
        d = ((a - s + 180) % 360) - 180          # signed offset from divider j (deg)
        dist = abs(math.sin(math.radians(d))) * r if abs(d) <= 90 else r   # a divider is a ray from the centre
        if best is None or dist < best[0]:
            best = (dist, j, d)
    dist, j, d = best
    here, other = (j, (j - 1) % n) if d >= 0 else ((j - 1) % n, j)
    return (other, dist) if here == k else (None, None)


def assign_candidates(dets, sec, trackers, p, spokes):
    """Blobs -> chambers. By position, except that a blob close to a divider goes to the chamber that
    was following a bug nearby (a bug walking along a divider must not change chamber every frame)."""
    h, w = sec.shape
    by_sec = [[] for _ in trackers]
    for d in dets:
        k = int(sec[min(max(int(d[1]), 0), h - 1), min(max(int(d[0]), 0), w - 1)])
        if k < 0:
            continue
        other, dist = border_info(d[0], d[1], p, spokes, k)
        if other is not None and dist <= p.border_px:
            mine, theirs = trackers[k].claim_dist(d), trackers[other].claim_dist(d)
            if theirs is not None and (mine is None or theirs < mine):
                k = other
        by_sec[k].append(d)
    return by_sec


# ------------------------------------------------------------ state logic
def _label(side, visible):
    if side == "A":
        return STATE_A if visible else STATE_A_HID
    return STATE_B if visible else STATE_B_HID


class StateMachine:
    """A/B state of ONE chamber, frame by frame. A bug does not flip sides every second, so:

    * evidence per frame: visible well inside A -> A, visible well outside -> B, visible within
      `hyst_px` of the boundary -> no evidence. Not visible (after `min_hidden_s`): last seen near
      the roof edge or in B -> B, last seen deeper in A (divider shadow) -> A, but only for
      `max_shadow_s` (the shadow bands lead to the roof: a long absence means the bug left).
    * the side only changes after the opposite evidence has lasted `min_dwell_s`; the switch is
      then dated back to when that evidence began (so latencies are not delayed).
    * before the bug is first seen it is in the shelter (B hidden).
    """

    def __init__(self, dt, p):
        self.p = p
        self.r_ab = p.boundary()
        self.min_f = max(1, int(round(p.min_hidden_s / dt)))
        self.dwell_f = max(1, int(round(p.min_dwell_s / dt)))
        self.max_shadow_f = max(self.min_f + 1, int(round(p.max_shadow_s / dt)))
        self.side = None
        self.sides, self.vis = [], []
        self.last_r, self.run = None, 0
        self.cand, self.cand_start = 0, 0

    def _evidence(self, o):
        p = self.p
        if o is not None:
            if self.last_r < self.r_ab - p.hyst_px:
                return "A"
            if self.last_r > self.r_ab + p.hyst_px:
                return "B"
            return None
        if self.run < self.min_f:
            return None
        if self.side == "B" or self.last_r is None or self.last_r >= self.r_ab - p.entry_zone_px:
            return "B"
        return "A" if self.run < self.max_shadow_f else "B"

    def update(self, o):
        i = len(self.sides)
        if o is not None:
            self.run = 0
            self.last_r = math.hypot(o[0] - self.p.cx, o[1] - self.p.cy)
            if self.side is None:
                self.side = "A" if self.last_r < self.r_ab else "B"
        else:
            self.run += 1
        if self.side is None:
            self.sides.append("B")
            self.vis.append(False)
            return STATE_B_HID
        ev = self._evidence(o)
        if ev == self.side or (ev is None and o is not None):   # seen near the boundary: still in play
            self.cand = 0
        elif ev is not None:
            if self.cand == 0:
                self.cand_start = i
            self.cand += 1
            if self.cand >= self.dwell_f:
                self.side = ev
                for j in range(self.cand_start, i):
                    self.sides[j] = ev
                self.cand = 0
        self.sides.append(self.side)
        self.vis.append(o is not None)
        return _label(self.side, o is not None)

    def labels(self):
        return [_label(s, v) for s, v in zip(self.sides, self.vis)]


def classify(obs, dt, p):
    """obs: per processed frame, (x, y) or None for ONE chamber -> list of states."""
    m = StateMachine(dt, p)
    for o in obs:
        m.update(o)
    return m.labels()


def summarise(states, obs, dt, p):
    n = len(states)
    total = n * dt
    tA = sum(s in IN_A for s in states) * dt
    first_A = next((i for i, s in enumerate(states) if s == STATE_A), None)
    grp = [s in IN_A for s in states]
    crossings = sum(1 for a, b in zip(grp, grp[1:]) if a != b)
    dist, prev, gap = 0.0, None, 0
    for o in obs:
        if o is None:
            gap += 1
            continue
        if prev is not None and gap * dt <= 2.0:
            d = math.hypot(o[0] - prev[0], o[1] - prev[1])
            dist += d if d >= 2.0 else 0.0   # ignore centroid jitter
        prev, gap = o, 0
    ppc = p.px_per_cm()
    return dict(
        time_A_s=round(tA, 2), time_B_s=round(total - tA, 2),
        prop_A=round(tA / total, 4) if total else float("nan"),
        latency_A_s=round(first_A * dt, 2) if first_A is not None else round(total, 2),
        censored=first_A is None, n_crossings=crossings,
        distance_cm=round(dist / ppc, 1),
        time_A_visible_s=round(sum(s == STATE_A for s in states) * dt, 2),
        time_A_hidden_s=round(sum(s == STATE_A_HID for s in states) * dt, 2),
        time_B_visible_s=round(sum(s == STATE_B for s in states) * dt, 2),
        time_B_hidden_s=round(sum(s == STATE_B_HID for s in states) * dt, 2),
        pct_frames_visible=round(100.0 * sum(o is not None for o in obs) / n, 1) if n else 0.0,
        trial_duration_min=round(total / 60, 2),
    )


# ------------------------------------------------------------------ driver
class Cancelled(Exception):
    pass


def run_analysis(video, p, out_dir, progress=None, cancel=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    p = replace(p)                 # the bug area measured below must not leak into the caller's settings
    cv2.setRNGSeed(12345)          # k-means blob splitting: same video -> same result
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_start = int(p.start_s * fps)
    if p.empty_bg:
        bg = load_empty_background(p.empty_bg, (h, w))
        log(f"background: empty arena from {os.path.basename(p.empty_bg)}")
    elif p.empty_first_s > 0:
        bg = median_background(cap, 0, int(p.empty_first_s * fps))
        f_start = max(f_start, int(p.empty_first_s * fps))
        log(f"background: first {p.empty_first_s:g}s of the video (analysis starts after it)")
    else:
        bg = median_background(cap, f_start, nf, 60)
        log("background: median of the whole video (no empty-arena footage; a bug that never "
            "moves may be absorbed -- record the empty arena first if possible)")
    f_end = nf if not p.max_minutes else min(nf, f_start + int(p.max_minutes * 60 * fps))

    spokes = find_spoke_angles(bg, p)
    sec = build_sector_map(bg.shape, p, spokes)
    roi = sec >= 0
    log("divider angles: " + ", ".join(f"{a:.0f}" for a in spokes))
    if not p.bug_area:
        p.bug_area = None
        est = estimate_bug_area(cap, f_start, f_end, bg, roi, None, p)
        p.bug_area = est
        log(f"one bug = {est:.0f} px (blobs much bigger than this are split)" if est else
            "could not measure the size of one bug; merged bugs will not be split")
    cv2.imwrite(os.path.join(out_dir, "calibration.png"), draw_overlay(bg, p, spokes, sec))

    n = p.n_sectors
    dt = p.step / fps
    obs = [[] for _ in range(n)]
    trackers = [ChamberTracker(dt, p) for _ in range(n)]
    machines = [StateMachine(dt, p) for _ in range(n)]
    times = []
    vw = None
    if p.save_video:
        sc = 0.6
        vw = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / p.step, (int(w * sc), int(h * sc)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    f = f_start
    total_proc = max(1, (f_end - f_start) // p.step)
    done = 0
    live_states = [STATE_B_HID] * n
    while f < f_end:
        if cancel and cancel():
            raise Cancelled()
        ok = cap.grab()
        if not ok:
            break
        if (f - f_start) % p.step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = _gray(frame)
            dets = detect(gray, bg, roi, p)
            by_sec = assign_candidates(dets, sec, trackers, p, spokes)
            for k in range(n):
                obs[k].append(trackers[k].update(by_sec[k]))
                for j, pos in enumerate(reversed(trackers[k].backfill), start=2):
                    obs[k][-j] = pos          # the frames that had to be confirmed first
            times.append((f - f_start) / fps)
            done += 1
            if vw is None:
                for k in range(n):
                    machines[k].update(obs[k][-1])
            if vw is not None:
                # online state for the check video; the final numbers use the same rules, but dated back
                img = draw_overlay(frame, p, spokes, labels=False)
                for k in range(n):
                    live_states[k] = machines[k].update(obs[k][-1])
                    o = obs[k][-1]
                    for d in by_sec[k]:   # grey dots: other blobs that were NOT chosen as the bug
                        if o is None or math.hypot(d[0] - o[0], d[1] - o[1]) > 3:
                            cv2.circle(img, (int(d[0]), int(d[1])), 5, (170, 170, 170), 1)
                    col = {STATE_A: (0, 220, 0), STATE_A_HID: (0, 160, 255),
                           STATE_B: (255, 120, 0), STATE_B_HID: (200, 0, 200)}[live_states[k]]
                    if o is not None:
                        cv2.circle(img, (int(o[0]), int(o[1])), 14, col, 2)
                    a0 = spokes[k]
                    span = (spokes[(k + 1) % n] - a0) % 360
                    t = math.radians(a0 + span / 2)
                    rr = p.r_out + 18
                    cv2.putText(img, f"{chamber_label(k, p)}:{live_states[k]}",
                                (int(p.cx + rr * math.cos(t)) - 40, int(p.cy + rr * math.sin(t))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.putText(img, f"{times[-1] / 60:5.1f} min", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                vw.write(cv2.resize(img, None, fx=0.6, fy=0.6))
            if progress and done % 50 == 0:
                progress(min(1.0, done / total_proc))
        f += 1
    cap.release()
    if vw is not None:
        vw.release()
    if progress:
        progress(1.0)

    # ---- final classification, summaries, frame table
    rows, frame_rows = [], []
    for k in range(n):
        st = classify(obs[k], dt, p)
        s = summarise(st, obs[k], dt, p)
        s = dict(chamber=chamber_label(k, p), **s)
        rows.append(s)
        for t, o, state in zip(times, obs[k], st):
            frame_rows.append((chamber_label(k, p), f"{t:.2f}", "" if o is None else f"{o[0]:.1f}",
                               "" if o is None else f"{o[1]:.1f}", state))
    rows.sort(key=lambda r: r["chamber"])
    main_cols = ["chamber", "time_A_s", "time_B_s", "prop_A", "latency_A_s", "n_crossings", "distance_cm"]
    with open(os.path.join(out_dir, "summary_per_chamber.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(main_cols)
        for r in rows:
            # latency is only meaningful if the bug ever entered A; otherwise NA
            wr.writerow([("NA" if (c == "latency_A_s" and r["censored"]) else r[c]) for c in main_cols])
    with open(os.path.join(out_dir, "summary_details.csv"), "w", newline="") as fh:   # QC only
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(out_dir, "frames_states.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["chamber", "time_s", "x", "y", "state"])
        wr.writerows(frame_rows)

    # trajectories on the calibration picture
    img = draw_overlay(bg, p, spokes, labels=True)
    pal = [(0, 0, 255), (255, 0, 0), (0, 160, 0), (0, 128, 255), (200, 0, 200), (0, 200, 200), (128, 128, 0), (0, 0, 128)]
    for k in range(n):
        pts = np.array([o for o in obs[k] if o is not None], np.int32)
        if len(pts) > 1:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, pal[k % len(pal)], 1)
    cv2.imwrite(os.path.join(out_dir, "trajectories.png"), img)
    log(f"done -> {out_dir}")
    return rows
