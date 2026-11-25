# Known issues

**Every output of this project is a simulation result. Nothing is run on, or validated on, the real UR10e; no output is
meant to drive it.**

The open limitations and accepted risks of the outputs: only what was decided or measured. Sections are cited as
SPEC §n (`SPEC.md`), which also states the reasons for the main choices.

## The legacy inputs' t_f column holds the catch heights

The earlier work's `save_data.py` wrote `t_f` from the `r_n_Z` column, so the stored flight times of
`Data/Legacy/Experiment_1_*.txt` are 0.39–1.07 s (the catch heights), not the drawn 0.75–2.25 s (legacy `0e59db2`).
- **Impact:** the legacy throws are shorter and flatter than intended; they serve the port's equivalence only (§4.1).
- **What would close it:** nothing; the inputs cannot be regenerated (the generator was unseeded).

## The baseline differs from the 2022 slide in two columns

Mean reproduces the 2022 slide exactly; Last is 5–9 % lower and Mean_Opt 3–6 % higher. The cause is not known.
- **Impact:** the slide is only a reference; the port is proven against the measured baseline (SPEC §4.1).
- **What would close it:** the slide's input files, if they exist.

## The recordings miss about half of the frames in flight

Only 19–97 % (p10–p90 29–69 %, median 44 %) of the frames inside a flight carry a sample, and a gap of 1 s or more
cannot be seen in the time column; hidden gaps are flagged by a continuity rule and a cited list of 18 steps.
- **Impact:** predictors must take irregular samples; no segmented flight contains a hidden gap, one borders one.
- **What would close it:** nothing available: the .qtm captures exist, but no marker data can be exported.

## The segmentation's acceptance can pass a segment joined across a long dropout

A synthetic throw whose ascent is hidden by a 0.67 s dropout is segmented with two hand samples joined to a truncated
descent (a_z −10.43 m/s², 1.15 m/s² horizontal pass the bounds); no real flight looks like this (a_z −9.65…−9.89).
- **Impact:** a new recording with such a dropout could give a wrong segment; check a_z against the measured ranges.
- **What would close it:** acceptance bounds tightened to the measured ranges, or a continuity check across dropouts.

## The flight model's residual is above the noise, and its cause is unknown

Quadratic drag with one k leaves 5.42 mm RMS (median) on the identification flights and 6.62 mm on the validation
flights, against a 3D noise of 1.24 mm; no further term improved held-out flights. The signs point to the measurement
(larger next to dropouts; a constant offset per present run removes 72 % of its sum of squares, 46–50 % shuffled).
- **Impact:** real-flight errors include this residual; the estimated floor (4.25 / 5.43 mm, SPEC §9.8) is a heuristic.
- **What would close it:** marker data, which are not available.

## The flight segments come from a constant-acceleration fit

A flight is cut where a constant-acceleration fit stops fitting within τ = 20 mm (SPEC §7.3); between 15 and 20 mm the
start moves in 55 of 90 flights by up to 80 ms. The first 40 ms raise the max residual from 16.8 to 23.0 mm.
- **Impact:** the identification drops the first 40 ms; k moves 0.009289–0.009426 1/m over τ 10–50 mm, near `Drag_Std`.
- **What would close it:** a segmentation cut with the identified model.

## The robot's position in the recordings is assumed

No hand-eye result exists; the rigid transform of SPEC §7.3 is chosen so the recorded flights descend through the catch
area. It keeps distances, so a prediction error on a real flight does not depend on it; the coverage does.
- **Impact:** every result on a real flight rests on this assumption, and says so; no catch metric uses real flights.
- **What would close it:** a measured robot position in the QTM frame.

## The scene's lab, camera positions, ball mass and accelerations are assumed

The 12 camera mounts and the rail (3.0 m, centred above (0, −3) m) are read from `Holder.stp` and assumed; the ball's
0.122 kg is not weighed (inertia a homogeneous sphere's); the cup's 0.476 kg is PLA at 1 240 kg/m³; the UR10e's
800 °/s² and 2500 mm/s² are UR's recommendation, not a published limit. Each is marked ASSUMED in the scene's build.
- **Impact:** half the cup's mass moves the catches by 0.4–1.4 points; a 0.100 or 0.150 kg ball by 0.5 points at 0.20.
- **What would close it:** a weighed ball and cup; a measured lab (SPEC §14).

## The assumed joint acceleration decides how much the robot reaches in time

The ball enters a 1.0 m sphere about the robot 0.76 s (5 %) / 0.99 s (median) after its 20th sample. Rest to rest within
0.7 s the cup reaches 0.947 m³ of catchable points at 800 °/s² and 0.370 m³ at 400 °/s² (SPEC §4.10).
- **Impact:** every statement about what the robot catches in time rests on 800 °/s²; results are given at 800 and 400.
- **What would close it:** a sourced acceleration limit of the UR10e, or a measured one.

## The catch area holds a tenth of the throws, and it lies beside the robot

`Areas.Catch` is 0.3 × 0.75 × 0.5 m at x 0.55…0.85, y −0.55…0.20, z 0.65…1.15 m. 9.7 % of `Default`'s train throws
pass through it; of those 96.5 % (800 °/s²) and 95.1 % (400 °/s²) are catchable inside it.
- **Impact:** a throw that misses the area is a no-go by construction; the search is heuristic, a larger area may exist.
- **What would close it:** a measured acceleration, other margins, or another shape (a union of cuboids).

## The catch area is checked at lattice points, and its reach is rest to rest

A point qualifies when a catch pose exists there at rest; between the 0.05 m lattice points nothing is checked. The
area's reach times and the decision's time test are rest to rest from the ready pose for both catches.
- **Impact:** a go may get a slower stroke or none (the static plan): re-plans start from a moving robot.
- **What would close it:** a decision that knows the stroke (not planned).

## A moving catch's stroke is planned once, on a joint line, with an assumed compute budget

The stroke is a straight line in joint space through the catch pose, started 0.195 / 0.202 s before the catch time
(test) and not re-planned once it runs; the lateral miss at the mouth is 13.7 / 16.6 mm against the static
10.7 / 11.3 mm (medians, 800 / 400 °/s²), while 99.3 / 98.5 % of `Catch_In`'s goes are retained against 97.0 / 96.6 %.
- **Impact:** the rates hold for the assumed 0.10 s budget; the cup peaks at 1.5 m/s (median), no TCP speed cap applies.
- **What would close it:** a stroke corrected while it runs, or a continuous motion without the stop.

## The ball's restitution is assumed, and the catch stands or falls with it

The nominal restitution 0.20 is an estimate for the real ball; there is no measured source. The static catch retains
94.4–100 % of the goes at 0.20 and 26.6–32.5 % at 0.29 (validation); the moving catch moves the half-way point from
about 0.26 to about 0.48 / 0.39 (800 / 400 °/s²). The recordings' 0.72 is the ball on the lab's mattresses, not used.
- **Impact:** every catch rate holds for the assumed contact only; friction and masses move it by 2–4 points at 0.20.
- **What would close it:** a drop test of the real ball onto the cup's material.

## The contact's restitution depends on the speed, and a bounce scatters with the step

The nominal contact returns 0.19 of the arrival speed at 7 and 3 m/s and 0.16 at 1 m/s; a single bounce scatters by
±0.012 with the contact's phase within the 0.00025 s step; the penetration is 8.4 mm at 7 m/s.
- **Impact:** levels are named by their restitution at 7 m/s; low-speed bounces in the cup are softer than the name.
- **What would close it:** a smaller step (twice the cost per halving), or a contact law with its own restitution.

## One seed retains fewer of its goes than the others, and the cause is not named

Seed 4 of `QT_BASIS` retains 95.2 % of its validation goes (test 95.4 %) against 96.9–97.7 % (97.0–97.5 %): its balls
pass the mouth plane a little farther from the axis and bounce out more often from the wall.
- **Impact:** a retained share is quoted as the range over the 5 seeds (a spread of 2.5 points on validation).
- **What would close it:** the ball's velocity against the cup's axis at the entry, per go (not planned).

## The robustness sweep's error is a construction, harsher than a predictor's

The sweep replaces the prediction by the truth plus an error sized from the stored real-flight error norms; direction,
links between ranges and re-plans, and everything beyond N 40 are assumed. With the simulated errors it retains 3.3–4.1
points less than `QT_BASIS` itself; at the real level two throws go falsely (1 of 789, 1 of 1 340 not catchable).
- **Impact:** a what-if on simulated throws; the stored real-flight errors never reach the decision's margins.
- **What would close it:** per-sample real prediction errors with their direction.

## The path check covers the robot, not the ball

The interception's path check holds the robot against the stand, the floor and itself; the ball is not one of its pairs.
- **Impact:** a kinematic interception is not a catch; the physical catch classifies a ball meeting a link (none seen).
- **What would close it:** nothing planned; the planner does not avoid the ball's path.

## The MuJoCo ball agrees with the flight model in free flight only

The scene's ball is within 0.98 µm and 1.39 µm/s of `Flight.Core.Simulate` to the floor (SPEC §4.13); in contact the
read velocity is off by up to h/2 times the contact's acceleration and MuJoCo's step is first order.
- **Impact:** no agreement is claimed after the first contact (1.2 mm/s for g at the catch's step, against 0.05 m/s).
- **What would close it:** nothing is planned.

## A ball spinning after a contact flies without a Magnus force

The flight model has no spin term: a ball leaving the rim, a link or the stand keeps its spin, its flight is not bent.
- **Impact:** matters only after a first contact (bounced off, a second chance at the cup).
- **What would close it:** a measured spin after a contact; not planned.

## Some throws reach the stand or the robot before the floor

Of 300 `Default` throws flown in the full scene, the first contact was the floor in 205, the stand in 57, the robot
in 35 and the cup's outside in 3.
- **Impact:** for a go these are classes (`hit_stand`, `hit_robot`); a no-go is not simulated, its contacts are unknown.
- **What would close it:** simulating the no-go throws too (not planned).

## The late re-plans rest on QT_BASIS extrapolating beyond N 40

`QT_BASIS` was trained on prefixes of N 10 to 40; the interception gives it prefixes of several hundred samples. Its
error keeps falling with N on simulated flights (lateral miss 14.5 mm 0.6 s before the catch, 8.8 mm at 0.05 s).
- **Impact:** the late re-plans' precision rests on that extrapolation; on real flights it is not measured.
- **What would close it:** a predictor trained on the longer prefixes.

## The motion-capture latency, the compute budget and the controller are assumed

The latency is 4 ms, a static plan's computation 0.05 s and a moving one's 0.10 s, the control one 2 ms tick with
position servos; the real pipeline and controller are not modelled. The planner is slower than its budgets: a moving
plan takes 117 / 152 ms at p95, a static one 57 / 58 ms (one process, one thread). GLS_M4's prediction alone takes
45–133 ms, `QT_BASIS`'s 0.5–2.4 ms, which is why the catch runs use `QT_BASIS`; GLS_M4's catch rate was never measured.
- **Impact:** every time margin holds for these values; a real UR10e would need its own safety limits (not certified).
- **What would close it:** a faster planner with the same outputs; a measured latency and controller (out of scope).

## The 2022 throw detector's direction branch never fires

The detector expects a thrower at −y while the 2022 areas put it at +y, so only its area branch fires: 20 to 155 samples
after the release on all 12 000 legacy runs. The test reference keeps it as it is.
- **Impact:** the test reference only; the 2022 pipeline is not a project predictor.
- **What would close it:** nothing needed.

## Simulated dropouts are composed from 45 recorded patterns, and depend on time, not on height

Every simulated pattern is composed of the 45 identification flights' patterns, replayed by the time since the release;
real presence depends on height (nothing is seen above z = 2.75 m). 370 Hz has no recording (300 Hz replayed).
- **Impact:** high simulated throws are observed more densely than real ones; knowing the patterns gave no gain.
- **What would close it:** a visibility model with the camera positions; new recordings.

## The configured error model leaves 10 of the 19 error statistics outside their bounds

Against the identification flights' bootstrap bounds (the median of 5 synthetic sets, SPEC §9.9), 10 of 19 statistics
lie outside: the residual 7.155 against 5.424 mm (bound 5.085–5.578), rougher within runs at 5–40 ms, larger offsets.
- **Impact:** simulated observations are rougher than real ones, the conservative direction for training.
- **What would close it:** a revised error model; no bound is widened.

## The error model's parameters are only partly identified by 45 flights

The estimator is deterministic but its estimates spread with the data: replicate studies identify the within-run
structure (within ±15 %), not the white noise, the offset's size and time constant apart, or the degrees of freedom.
- **Impact:** `Observation.*` are effective calibration values, one estimate among a spread (SPEC §7.5).
- **What would close it:** more recorded flights, or marker data.

## The generated throws stay in the air a little longer, and more of them go above the cameras

11 of the 12 throw-shape statistics lie inside their bounds; the floor-time median does not (1.2746 s against
[1.2325, 1.2701] s). 28.5 % of the generated throws go above z = 2.75 m against 20.5 % of the real releases.
- **Impact:** the median generated flight lands 22 ms later; high throws get lower flights' dropout patterns.
- **What would close it:** a flight-time scheme that also matches the floor time, or new recordings.

## Three identification flights lie outside the support of the generated throws

42 of the 45 identification flights lie inside the support of `Throw.Target_Area`; two cross only the far side, one
starts late.
- **Impact:** flights like the two are not represented in the simulated training data.
- **What would close it:** targets on the far side too (the whole hemisphere put 4 of 12 shape statistics outside).

## The decision's truth credits catches one lattice step beyond the area it aims into

The truth counts a throw catchable anywhere in `Areas.Catch` grown by 0.05 m, while the decision goes only for a
predicted point inside `Areas.Catch` shrunk by its margins.
- **Impact:** a go may send the robot to a point inside the area for a ball up to 0.05 m beyond it.
- **What would close it:** the physical catch decides; the decision itself is unchanged.

## The catch throws are faster and flatter than the real ones, and few real throws pass through the catch area

The catch datasets' shape lies outside the identification bounds in 3–5 of 12 statistics (release speed 6.94 m/s
against 6.64–6.81); by the assumed transform 1 of 45 identification flights passes through `Areas.Catch`.
- **Impact:** the decision's rates hold for throws aimed into the area, which the recordings do not show.
- **What would close it:** recordings of throws aimed at the catch area, or a measured transform.

## Truth on a real flight is the recording itself

The error is measured against the recorded samples after the prefix, with their noise (0.61 mm median) and dropouts of
up to 0.975 s.
- **Impact:** real-flight errors include the noise, reported in horizon ranges with their flight counts.
- **What would close it:** nothing in the recordings.

## A dataset is reproducible only with the same NumPy version and platform

NumPy's random distributions may change between versions (NEP 19); byte-for-byte reproduction needs the same code,
NumPy and platform.
- **Impact:** a NumPy upgrade gives new dataset hashes, and old runs refuse the new datasets.
- **What would close it:** nothing; the manifest records the versions.

## Real-flight truth is sparse at long horizons, and N is a poor measure of what was observed

At N = 20, 17 of 45 real flights have a sample in [1.2, 2.0] s; at N = 40, 2 do. N = 40 spans 0.195–0.984 s (p10–p90).
- **Impact:** long-horizon statistics rest on few flights; results are also given by prefix duration.
- **What would close it:** new recordings with better in-flight coverage.

## The floor is taken as the recordings' z = 0, and the stand as the CAD's

Trajectories end at z = 0.030 m (the ball's radius); the mounting face is 0.180 m above the floor as in the CAD. The
recorded flights land about 30 mm higher (63 mm crossing, 19 flights): the lab floor was covered with mattresses.
- **Impact:** the end mismatch and the predicted end's time on real flights move with it; evaluated samples do not.
- **What would close it:** a measured base height and floor in the recordings' frame.

## The scene's collision geometry is simplified

The robot collides as the vendored model's capsules, the stand as four boxes, the cup for the ball as a bottom, 24
staves and 24 lip boxes; the ball as its sphere without its markers. The real ball measures 91.9 mm over its markers
against a cup bottom 81.5 mm wide. With the markers as rigid shapes (a variant) more balls are retained at high
restitution.
- **Impact:** clearances are conservative for the stand and the cup; how a marked ball rests in it is not simulated.
- **What would close it:** a measured ball (its markers' stiffness, its rebound in the cup).

## The bounded error model is milder than the recordings at the extreme

M4's offset and white noise are bounded at 0.070 and 0.040 m; the largest residual reaches the real 55.9 mm in 17 of 100
synthetic sets (55 unbounded).
- **Impact:** a flight like the worst recorded one is rare in the simulated data; GLS_M4 uses the unbounded variances.
- **What would close it:** a revised error model whose tails are estimated with their bound, or more flights.

## The numerical predictors' error is the observation error carried forward

On simulated validation GLS_M4's median remainder RMS is 57.2 / 34.7 / 20.5 mm at N = 10 / 20 / 40, already 10–12 mm
at 0.05 s: M4's per-run offset cannot be told from the position.
- **Impact:** simulated errors are as rough as M4, rougher than the real observations.
- **What would close it:** a revised error model changes the numbers, not the predictors.

## Three real validation flights lie outside the support of the generated throws

42 of the 45 real validation flights lie inside the support (two releases outside `Areas.Throw`, one never in the
target area).
- **Impact:** every real-flight result is reported for all 45 and for the 42 inside; three are too few for statistics.
- **What would close it:** nothing planned; the generator is designed on the identification flights only.

## The learned predictors are level with the numerical ones, on simulated and on real flights, not better

On the simulated test partition `QT_BASIS` has 59.7 / 36.3 / 22.4 mm against GLS_M4's 59.2 / 35.9 / 21.1 mm; on the
45 real flights the paired difference is −0.29 [−0.72, +0.26] / +0.45 [−0.71, +1.08] / −0.23 [−0.94, +1.30] mm.
- **Impact:** a learned predictor adds speed (0.4–0.9 ms against 24–50 ms per trajectory), not accuracy.
- **What would close it:** a better observation model would change what there is to learn.

## A learned predictor keeps an error of its own on error-free observations

On error-free observations GLS_M4 is exact, while `QT_BASIS` trained on five noise scales keeps 1.3 / 0.8 / 0.9 mm.
- **Impact:** small against the 22–60 mm of the observation error.
- **What would close it:** nothing planned (twice the optimiser steps: 1.0 / 0.6 / 0.8 mm for twice the training).

## The median of 45 real flights moves by 10 mm with one flight's rank at N = 10

At N = 10 the 45 values have a gap at their middle (GLS_M4's ranks 22–24: 43.7, 52.1, 66.1 mm), so `QT_BASIS`'s
median reads 61.9 against 52.1 mm while per flight they differ by −0.29 mm.
- **Impact:** a difference of two medians at N = 10 is not a difference between predictors; the paired one is.
- **What would close it:** more recorded flights.

## A query-time trajectory is not a flight by construction

The query-time model's trajectory is a learned function of time; a prefix whose trajectory starts below the floor or
never lands within 2.0 s is refused as its own outcome. No real flight was refused (0 of 4 921 predictions).
- **Impact:** a refusal is a no-go in the decision, so it never moves the robot.
- **What would close it:** nothing planned.

## The numerical predictor spends its time integrating to the floor

A numerical trajectory takes 25–52 ms, about 15 ms of it RK4 to the floor in Python; re-planning comes every 3.3–10 ms.
- **Impact:** only the query-time model answers within the re-planning period.
- **What would close it:** a faster integration to the floor with the same outputs.

## The view shows more than the catch computed, and its window is checked by eye

`scripts/catch.py view` plays back a throw computed by the code of `catch`, but the compared methods' lines, the
overlay's inference time, the "Throw Detection" mark (the first decision sample) and a no-go's ball are not results.
- **Impact:** a picture from `view` shows the stored result only for the driving method of a stored validation throw.
- **What would close it:** a renderer with soft shadows; the area's reach read from a stored map (start-up 23 s).

## The figures rest on a style library that uses two deprecated functions, and the stored statistics they draw are limited

SciencePlots 2.2.2 calls two functions matplotlib 3.13 will remove (silenced at its import). The figures draw only what
is stored: no interval of a median, no quartiles, the 90th percentile as the whisker. The restitution axis is the
measured mean over 8 phases; the real-flight figures show identification flights (demonstrations, not an evaluation).
In the light theme the "no go" grey is held at the 4.5:1 contrast floor, so it cannot be lighter.
- **Impact:** a later matplotlib may break the style's import; a cliff's second decimal is not a measurement.
- **What would close it:** a SciencePlots release without the calls; quartiles written by `evaluate` and `compare`.
