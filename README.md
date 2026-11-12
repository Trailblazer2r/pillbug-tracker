# Pill bug light / shelter tracker

Tracks one focal pill bug per chamber in a circular arena split into radial chambers
(centre cup + lit open area **A** + roofed shelter **B**) and reports, for each bug,
how long it spent in A vs B.

## Run

```bash
pip3 install -r requirements.txt
python3 pillbug_gui.py
```

1. Choose the **video**. The cup circle, the roof's inner-edge circle and the number of chambers are
   detected automatically. Optionally also choose a short image/video of the **empty arena**.
2. Check the preview and, if needed, **drag** the circles: the edge of the inner red circle (cup),
   the edge of the outer red/yellow circle (roof inner edge), or anywhere else to move both.
   The blue band is B (under the black roof): it is **not tracked**, a bug that goes there simply disappears.
   Everything else (chamber count, thresholds, ...) is under "Advanced settings" and normally needs no change.
3. **Run analysis**. Outputs go to the output folder:

| file | content |
|---|---|
| `summary_per_chamber.csv` | one row per bug: `time_A_s, time_B_s, prop_A` (main result), `latency_A_s` (time until first in A; NA if never), `n_crossings` (A<->B switches), `distance_cm` |
| `summary_details.csv` | QC only: visible/hidden split of the times, % frames detected, censored flag |
| `frames_states.csv` | state of every chamber at every analysed frame |
| `calibration.png`, `trajectories.png` | geometry check and trajectories |
| `annotated.mp4` | check video: circle colour = state (green A, orange A hidden, blue B, purple B hidden) |

## How "not visible" is interpreted

Identity = chamber (the bug seen in chamber *k* is chamber *k*'s bug).
The roof over B hides the bug from the overhead camera, and the dividers cast shadows, so a bug
can disappear in two very different ways:

* last seen **deep inside A** (e.g. in a divider's shadow) -> still **A** (`A_hidden`), not shelter
* last seen **near the roof edge (within the entry zone, default 45 px) or in B** -> **B** (`B_hidden`)
* before the bug is first seen (it starts under the cup in B) -> `B_hidden`
* dropouts shorter than "min hidden time" keep the previous side

A bug does not switch sides every second, so a switch between A and B is only accepted after the new side has held for
**2 s** (`min_dwell_s`; the switch is dated back to when it started), and a bug seen within 12 px (`hyst_px`) of the
boundary counts as "no change".

## How the bug is found

* the background difference uses a gain (not an offset), so dark places (dividers, roof shadow) do not light up when the camera brightness drifts
* blobs without a strong core (`min_contrast`) and long thin ones (shadow edges) are ignored
* bugs of neighbouring chambers that touch are separated along the divider lines; a blob much larger than one bug
  (`bug_area`, measured from the video) is split into several bugs and the one closest to the previous position is followed
* a detection that jumps farther than a bug can walk is only believed after it stays put for 1 s (`confirm_s`)
* `annotated.mp4`: grey dots are blobs that were seen but NOT chosen as the bug

`time_A_hidden_s` etc. are reported so you can see how much of the data relies on this rule.
Unit tests for the rule: `python3 -m unittest test_states`.

## Notes / limits

* Without empty-arena footage the background is the median of the whole video, which can absorb a bug
  that never moves. Record 10-20 s of the empty arena (or leave the first N s empty and set
  "First N s are empty").
* Distance is converted to cm using the centre-cup diameter (default 5 cm).
* The entry zone was set from one 30-min pilot (disappearances start either at the cup wall or at the roof
  edge, never in between). Re-check `annotated.mp4` on new runs, especially bugs that vanish mid-arena.
