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
from dataclasses import dataclass, field
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
def detect(gray, bg, roi, p):
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.int16)
    diff = g - b
    diff -= int(np.median(diff[roi]))   # cancel global exposure drift
    m = ((np.abs(diff) > p.thresh) & roi).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    return [(float(cents[i][0]), float(cents[i][1]), int(stats[i, cv2.CC_STAT_AREA]))
            for i in range(1, n) if p.min_area <= stats[i, cv2.CC_STAT_AREA] <= p.max_area]


def pick_for_chamber(cands, last):
    """One bug per chamber: keep the candidate closest to where the bug was, else the biggest."""
    if not cands:
        return None
    if last is not None:
        return min(cands, key=lambda c: math.hypot(c[0] - last[0], c[1] - last[1]))
    return max(cands, key=lambda c: c[2])


# ------------------------------------------------------------ state logic
def classify(obs, dt, p):
    """obs: per processed frame, (x, y) or None for ONE chamber -> list of states.

    Visible: A or B by radius. Not visible (shelter roof OR divider shadow):
      * short dropout (< min_hidden_s): keep previous side
      * last seen in/at the A/B boundary zone or in B, or never seen yet: B_hidden
      * last seen deeper in A: A_hidden (shadow / occlusion, NOT shelter)
    """
    r_ab = p.boundary()
    min_f = max(1, int(round(p.min_hidden_s / dt)))
    states = []
    prev, last_r, run = STATE_B_HID, None, 0
    for o in obs:
        if o is not None:
            run = 0
            last_r = math.hypot(o[0] - p.cx, o[1] - p.cy)
            st = STATE_A if last_r < r_ab else STATE_B
        else:
            run += 1
            if run < min_f:
                st = STATE_A_HID if prev in IN_A else STATE_B_HID
            elif prev in (STATE_B, STATE_B_HID) or last_r is None or last_r >= r_ab - p.entry_zone_px:
                st = STATE_B_HID
            else:
                st = STATE_A_HID
        states.append(st)
        prev = st
    return states


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
    cv2.imwrite(os.path.join(out_dir, "calibration.png"), draw_overlay(bg, p, spokes, sec))

    n = p.n_sectors
    dt = p.step / fps
    obs = [[] for _ in range(n)]
    last = [None] * n
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
            by_sec = [[] for _ in range(n)]
            for d in dets:
                k = int(sec[min(max(int(d[1]), 0), h - 1), min(max(int(d[0]), 0), w - 1)])
                if k >= 0:
                    by_sec[k].append(d)
            for k in range(n):
                d = pick_for_chamber(by_sec[k], last[k])
                if d is None:
                    obs[k].append(None)
                else:
                    obs[k].append((d[0], d[1]))
                    last[k] = (d[0], d[1])
            times.append((f - f_start) / fps)
            done += 1
            if vw is not None:
                # cheap online state for the check video (final numbers use classify on the full sequence)
                img = draw_overlay(frame, p, spokes, labels=False)
                for k in range(n):
                    live_states[k] = classify(obs[k][-int(60 / dt):], dt, p)[-1]
                    o = obs[k][-1]
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
    with open(os.path.join(out_dir, "summary_per_chamber.csv"), "w", newline="") as fh:
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
