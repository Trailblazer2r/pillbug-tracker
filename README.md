# Pill bug light / shelter tracker

Tracks one focal pill bug per chamber in a circular arena split into radial chambers
(centre cup + lit open area **A** + roofed shelter **B**) and reports, for each bug,
how long it spent in A vs B.

## Run

```bash
pip3 install -r requirements.txt
python3 pillbug_gui.py
```

1. Pick the **video** (and, ideally, a short image/video of the **empty arena**).
2. Check the preview: chamber numbers, the centre cup, the outer wall and the yellow A/B boundary.
   Adjust centre / radii / number of chambers until the overlay fits. "equal area" sets
   `r_AB = sqrt((R^2 + a^2) / 2)`.
3. **Run analysis**. Outputs go to the output folder:

| file | content |
|---|---|
| `summary_per_chamber.csv` | one row per bug: `time_A_s, time_B_s, prop_A, latency_A_s, censored, n_crossings, distance_cm` + QC columns |
| `frames_states.csv` | state of every chamber at every analysed frame |
| `calibration.png`, `trajectories.png` | geometry check and trajectories |
| `annotated.mp4` | check video: circle colour = state (green A, orange A hidden, blue B, purple B hidden) |

## How "not visible" is interpreted

Identity = chamber (the bug seen in chamber *k* is chamber *k*'s bug).
The roof over B hides the bug from the overhead camera, and the dividers cast shadows, so a bug
can disappear in two very different ways:

* last seen **deep inside A** (e.g. in a divider's shadow) -> still **A** (`A_hidden`), not shelter
* last seen **at the A/B boundary (within the entry zone) or in B** -> **B** (`B_hidden`)
* before the bug is first seen (it starts under the cup in B) -> `B_hidden`
* dropouts shorter than "min hidden time" keep the previous side

`time_A_hidden_s` etc. are reported so you can see how much of the data relies on this rule.
Unit tests for the rule: `python3 -m unittest test_states`.

## Notes / limits

* Without empty-arena footage the background is the median of the whole video, which can absorb a bug
  that never moves. Record 10-20 s of the empty arena (or leave the first N s empty and set
  "First N s are empty").
* Distance is converted to cm with the arena diameter (default 20 cm).
* Parameters (threshold, entry zone, ...) were tuned on pilot videos without a roof: check
  `annotated.mp4` on a real roofed run before trusting the numbers.
