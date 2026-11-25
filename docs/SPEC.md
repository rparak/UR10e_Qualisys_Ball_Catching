# Robotic Ball Catching (MuJoCo) — Specification

The specification of release v1.0, condensed from the development specification: what the program does, the facts it
rests on, its formats and its configuration. Every number is as it was measured; none was changed in condensing. The
section numbers are stable, because the code cites them as "SPEC §n". Where a choice's reason matters for the code,
it is stated in its section; the open limitations are in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

**Every output is a simulation result; nothing is run on, or validated on, the real UR10e.**

A *measured* number names what it was measured on; *assumed* marks a value without a measured source. Unless stated,
measurements were made on a MacBook M2 Pro (macOS 26.6.2, Python 3.11, NumPy 2.4.6, MuJoCo 3.16.0).

## 1. Goal

A MuJoCo simulation, by Roman Parak, of a UR10e robot with a passive conical cup gripper that catches a thrown ball.
The ball's flight model is identified from 90 real Qualisys recordings (§4.2); random throws are generated from it;
the rest of a flight is predicted from its first N observed points (a numerical and a learned predictor, chosen by
measurement); a conservative decision says whether the ball can be caught; a time-optimal interception is planned
within the UR10e limits; and the catch is simulated with contact. Everything runs from the command line (§6); a throw
is watched with `scripts/catch.py view`, the stored results are drawn by `scripts/plots.py`. There is no application.

## 2. Scope

Each requirement is testable; its acceptance is named with it. **The features**, which the code cites as F<n>:

| F | Feature | F | Feature |
|---|---|---|---|
| F1 | Legacy port | F9 | Catch decision |
| F2 | Recording loader | F10 | Interception planner |
| F3 | Flight identification | F11 | Physical catch |
| F4 | Scene and workspace | F12 | Cancelled (no application is built) |
| F5 | MuJoCo ball | F13 | Release v1.0 |
| F6 | Observation model and dataset | F14 | Tuning and final evaluation |
| F7 | Numerical predictors and evaluation | F15 | Moving catch |
| F8 | Learned predictors | | |

The view and the figures (§2.11) were built as "the visualisation", in two deliveries.

1. **Legacy port.** The numerical code of the 2022 work (commit `0e59db2` of the development history: the flight
   model, four predictor-corrector integrators, the key state Last / Mean / Mean_Opt, throw detection, approximate and
   final catch prediction, the areas) is rewritten and reproduces the baseline of §4.1 throw by throw within 1e-9 m.
   The areas and the flight model are project code (`src/Geometry`, `src/Flight`); the 2022 pipeline on top of them is
   a frozen test reference, `test/Legacy/Reference.py`.
2. **Recording loader.** Reads every file of §9.2, rebuilds the time (§7.3), reports every gap, duplicate and
   irregular step with file and row, and segments the main free flight of each recording.
3. **Flight model.** a = g − k·|v|·v; a further term (spin, Magnus, a bias) only if it improves held-out flights: none
   did (§4.2). k is identified on the identification flights and validated on the disjoint validation flights.
4. **Scene.** Built from the CAD (§4.4): the floor and camera rail, the UR10e on its stand, the conical cup, the ball
   with its 6 markers, the 12 cameras (visual only), the four areas; no human. The origin is on the floor under the
   base axis, the thrower at −y, the mounting face at z 0.180 m (§14.6). Built once by
   `scripts/Maintenance/build_scene.py`, then only checked; the fingerprint committed (§7.10, §9.10). *Acceptance:* it
   builds from any working directory without a warning; `scripts/Maintenance/derive.py scene` exits 0.
5. **Ball in MuJoCo** follows the flight model: the drag as an external force before every step, the velocity placed
   half a step back (§7.10). *Bound:* 0.1 mm and 1 mm/s against `Flight.Core.Simulate` at every step to the floor.
   *Met:* at most 0.98 µm and 1.39 µm/s over the 10 000 train flights of `Default`, `Catch_In`, `Catch_Near` (§4.13).
6. **Observation model and dataset generator.** Random throws from the throw area into a target area (§7.6); the
   observation at 100 / 200 / 300 Hz (370 Hz optional, *assumed*) with the error model M4 and composed dropouts (§7.5),
   measured on the 45 identification flights only. Time stamps are capture times; the latency is the planner's.
7. **Predictors.** The program holds two: the numerical `GLS_M4` (§7.7) and the learned `QT_BASIS` (§7.8). `LS`,
   `QT_MLP` and `HYBRID` were measured (§4.6–4.9) and removed before v1.0. A predictor predicts **the whole remaining
   trajectory** (§2.13.13) from irregularly sampled input. All are evaluated on the same held-out simulated flights
   with every metric of §7.12, and on the 45 real validation flights with the metrics of §2.14 only; by N, prefix
   duration and rate; over at least 5 seeds; split by flight. Learned predictors are trained, tuned, selected and
   early-stopped on simulated flights only.
8. **Catchability decision** (F9) with its false-go and false-no-go rates on simulated throws only; when in doubt,
   the robot does not move: a false go starts the robot towards a ball it cannot catch, while a false no-go only
   loses one catch, so the margins err towards no-go. Go at the first of the stages N = 20, 40 whose prediction
   passes the cross-track and time margins at q 0.95; a refused prediction is no-go (§7.11). *Acceptance* (fixed
   before the counted run): on the `Catch_Near` test partition the Clopper–Pearson 95 % upper bound of the false-go
   rate ≤ 0.1 % for each of the 5 `QT_BASIS` seeds at 800 and 400 °/s². *Met* (§4.11).
9. **Interception planner** (F10): time-optimal within the joint limits (§4.3) and the *assumed* accelerations,
   collision-free by exact distances, re-planned as the prediction updates, the cup's axis along the ball's velocity,
   the latency counted. No TCP speed cap: a simulation-only choice; a real UR10e would need its own safety limits,
   which this project does not certify. A decision at every sample with a ladder of margins; one plan at a time
   (budget 0.05 s, latency 4 ms), accepted when the robot rests 0.03 s before the catch time on a path certified at
   2.5 mm; the result kinematic at the mouth plane (lateral miss ≤ 32.5 mm, angle ≤ 10°). *Acceptance:* R1, item 8's
   bound; R2, every motion within the limits, certified and at rest before the crossing. *Met* (§4.12): R1 0.079 /
   0.078 %; R2 on 60 102 goes. *The moving catch* (F15; `Interception.Catch_Mode` `moving`, shipped): rest to rest to
   a stroke's start; the stroke runs every joint on one straight joint line through the catch pose, the cup along the
   ball's direction, its speed peaking at the predicted contact with the cup's bottom, then braking at once (§7.9,
   §7.11). It meets the ball in motion so that the ball arrives with less speed relative to the cup and more balls
   stay in it: the restitution cliff moves from 0.20–0.29 to about 0.48 / 0.38 (§4.16); the scale 1, 0.75, 0.5 or
   0.25 of the largest speed, else the static plan; a moving plan takes 0.10 s (*assumed*). `static` stays runnable.
   *Acceptance:* R1, the decision's columns equal the static catch's rows; R2, every stroke within the limits,
   certified, at rest at its start, the crossing inside it; R3 descriptive; R4 (fixed before the counted read): (a)
   pooled over the seeds the moving catch retains at least the static share of the same goes, (b) every one of the 20
   cells (dataset × acceleration × seed) at least 97.0 %. *Met* (§4.16).
10. **Physical catch** (F11). *Caught* means **retained**: 2.5 s after the catch time (*assumed*) the ball's centre is
    below the mouth plane, above the flange and within 62.5 mm of the axis. *At rest* (below 0.05 m/s relative to the
    cup for 0.2 s, *assumed*) is secondary: it depends on the rolling friction, which has no source. Every throw gets
    one of nine classes, mapped to caught / bounced off / missed / late / outside the workspace (§7.11). The contact is
    *assumed* (restitution 0.20, the author's estimate): the static catch falls from 97–98 % of the goes at 0.20 to
    25–31 % at 0.29. The moving catch moves that cliff (§4.16) and retains 98.4–99.5 % of the test goes (static
    95.0–97.6 %).
11. **The view and the figures** (an application, planned first, is not built). `view` shows one simulated throw in
    MuJoCo's viewer or as frames (§7.13, §9.15), computed with the code of `catch` and played back; never a test throw;
    nothing counted. *Acceptance:* a stored validation throw has the stored interception and catch rows byte for byte; a
    test throw is refused before its file is opened; the theme changes visual fields only. `plots.py` draws the stored
    results (§7.14, §9.16). *Acceptance:* every plotted number from its source file; colours at 4.5:1 or more; the
    motion's records equal the stored rows. GLS_M4 drives no robot: its prediction takes 24–50 ms at N 10–40 (§4.8)
    and 45–133 ms at N 20–150 (one process, not idle), longer than the re-planning period of 3.3–10 ms; no GLS_M4 catch
    was run, and its catch rate is not known.
12. **README** with how the real data was recorded (§9.2). No IP address is written into the tree.
13. **Dataset and training workflow**, one command each (§9.5–9.7). The 45 real validation flights are read only by
    `evaluate --partition test` (13.9) and by `identify`'s validation (§9.8), which fits nothing on them; each read is
    counted in a ledger.
    1. **Dataset command.** `dataset --config` generates, observes and splits by flight. *Acceptance:* partitions
       disjoint by flight, sized by the largest-remainder rule.
    2. **Reproducibility.** The same configuration, seed, code, NumPy version and platform give the same bytes;
       `derive.py dataset-check DIR` compares (exit 0 identical, 1 differs).
    3. **Stable prefix.** A flight's stream depends on the seed and its index only: the first 1 000 flights of a 2 000
       dataset equal a 1 000 dataset.
    4. **Samples as observed.** Own time stamps and dropouts, never resampled; the truth at every frame.
    5. **Observation model from data**, the 45 identification flights only: each pattern statistic inside its 95 %
       bootstrap interval (§9.9); at 300 Hz all but the share of flights with a dropout over 0.375 s (0 real).
    6. **Real flights covered.** The validation release states in the support of the generated throws, counted and
       the outside listed, inside `evaluate --partition test` only: 42 of 45 (§4.9).
    7. **Training command.** `train --config`, one model per seed (≥ 5), train and validation partitions only;
       *acceptance:* a leakage test fails a run that opens the test partition or `Data/Recordings/`.
    8. **Validation command.** `evaluate --run DIR`, the only input to selection and tuning; the same leakage test.
    9. **Test command.** `evaluate --run DIR --partition test`, the simulated test partition and the 45 real flights;
       every call a ledger row (§9.7); a run from a dirty tree or with a changed dataset hash refused.
    10. **Manifests.** `manifest.json` per dataset, run and evaluation (sorted keys, no wall-clock time; hashes, seeds,
        git state, platform, versions); an edited configuration changes its hash.
    11. **Metrics as distributions** (§7.12): per flight, per seed (mean, median, p90, max), the spread over seeds.
        Real flights with their first 40 ms trimmed (in-hand samples: 22 mm off free flight at the first, 5 mm by
        20–30 ms). A horizon range under 10 flights (*assumed*) lists its values, no statistics.
    12. **Fine-tuning boundary.** Tuning is `train` + `evaluate` on the simulated validation partition; no real or test
        flight is used for training, selection, tuning or early stopping.
    13. **Trajectory output contract.** From the prefix only, a trajectory fixed before any query: t_0, t_end (the
        ball centre at z = 0.030 m, §14.6), the state (position [m], velocity [m/s], robot frame) at any times in
        [t_0, t_end]. *Acceptance:* an off-grid query answered; one outside refused, naming the time and the range;
        all times, a subset and the subset reversed identical; every trajectory ends at the floor height.
14. **The roles of the real flights.** *(R1)* Two roles, kept apart by the F2 split (§7.3): the estimation of the
    simulation models on the 45 identification flights (k = 0.00930496 1/m; the observation model), the evaluation of
    prediction on the 45 validation flights. *(R2)* On real flights only the position error versus horizon, the
    remainder RMS, the end mismatch and the inference time. *(R3)* The robot never catches a real flight. *(R4)* The
    simulated criteria unchanged. *Acceptance:* the catch columns are empty for real flights (§9.7).

## 3. Out of scope

Running on the real robot; ROS (only to record the data, §9.2); vision-based detection; reinforcement learning; a human
in the scene; new recordings; the ball's orientation; Linux and Windows (§5); training, tuning, selection or early
stopping on real flights; catching, or catch metrics on, a real flight (§2.14).

## 4. Measured facts this specification rests on

The commands are named as they were run; their current names are in §6. The numbers of `LS`, `QT_MLP` and `HYBRID` are
the record of their development. **The counted reads**: the test ledger holds 11 rows (F7: rows 1–2; F14: rows 3–11),
the decision ledger 2 (F9), the interception ledger 1 (F10), the catch ledger 2 (F11, F15); the ledgers are generated
data and not part of the repository.

### 4.1 Baseline of the earlier code (measured 2026-11-09)

The 2022 benchmark script, its method and key-state settings only changed, with Python 3.10.22, NumPy 1.22.4, pandas
1.4.4, Matplotlib 3.5.3; inputs `Data/Legacy/Experiment_1_<method>_DRAG_False.txt` (1000 throws each); 370 Hz, 20
input states, no drag, no noise.

| Method | Mean distance Last / Mean / Mean_Opt [mm] | Time per throw [ms] | 2022 slide (Progress_2, slide 6) |
|---|---|---|---|
| Midpoint | 7.644 / 2.383 / 0.750 | 4.65 / 4.59 / 5.09 | 8.07 / 2.38 / 0.71 mm; 33.2–35.8 ms |
| Heun | 7.691 / 2.383 / 0.773 | 4.70 / 4.78 / 4.98 | 7.94 / 2.38 / 0.73 mm; 38.2–41.7 ms |
| RK2 | 7.545 / 2.383 / 0.790 | 4.61 / 4.83 / 5.23 | 8.04 / 2.38 / 0.77 mm; 35.2–36.7 ms |
| RK4 | 7.378 / 2.383 / 0.772 | 8.61 / 10.5 / 10.9 | 8.12 / 2.38 / 0.75 mm; 49.8–53.3 ms |

The 12 per-throw files are the port's reference (§9.3). Mean reproduces the slide; Last is 5–9 % lower, Mean_Opt 3–6 %
higher, cause unknown. **This is integrator self-consistency, not catch accuracy** (the truth from the same integrator,
no drag, no noise). The 2022 code at its last commit does not run (a change of `Euclidean_Norm` on 2022-08-22); the
port reproduces the earlier commit. The stored `t_f` holds the catch heights (0.39–1.07 s), read as it is. The earlier
learned predictors (the author's presentations): FFNN 12.0 mm / 0.78 ms; later MAE 5.52 mm (LSTM), 5.42 mm (GRU);
not reproduced.

### 4.2 The real recordings (measured 2026-11-09/10 on all 90 files)

| Folder | Files | Samples per file | Step | Duration [s] | Present / elapsed |
|---|---|---|---|---|---|
| `Frequency_100_Hz_Active_Filtering_OFF` | 10 | 1000 (one 2500) | 10.000 ms | 12.7–33.2 | 0.59 |
| `Frequency_100_Hz_Active_Filtering_ON` | 10 | 750 | 10.000 ms | 8.4–14.1 | 0.63 |
| `Frequency_200_Hz_Active_Filtering_OFF` | 50 | 2000 / 2500 / 5000 | 5.000 ms | 10.8–31.6 | 0.72 |
| `Frequency_200_Hz_Active_Filtering_ON` | 10 | 1500 | 5.000 ms | 8.8–12.5 | 0.72 |
| `Frequency_300_Hz_Active_Filtering_OFF` | 10 | 2500 | 3.333 ms | 10.9–18.0 | 0.65 |

- 184 000 samples, header `Time,Data_X,Data_Y,Data_Z`. **Time** is ns within the second; every step a whole number of
  periods (worst 0.03 %): a missing frame is a dropped sample. 7 305 steps of ≥ 1.5 periods, the longest 995 ms. **A
  gap of 1 s or more is invisible in this column**: 18 ambiguous steps (13 by the continuity rule, 2 zero steps, 3
  cited), none inside a flight. One throw per file, from y ≈ 0…+1 m towards −y; nobody catches it. At rest 0.023 mm.
- **Flights** (90 of 90): flight time 1.16–1.37 s (p10–p90), release speed 6.2–7.6 m/s, range 4.1–5.9 m, apex
  2.32–2.97 m. **Only 19–97 % (median 44 %) of the frames in flight carry a sample.** Dropout runs median / p99 / max:
  100 Hz 20 / 586 / 750 ms, 200 Hz 10 / 699 / 975 ms, 300 Hz 6.7 / 512 / 640 ms; 35 of 90 flights hold one > 0.375 s.
- **Fits** (45 identification flights, first 40 ms trimmed): **k = 0.009305 1/m**, bootstrap sd 0.000157 (2 000
  resamples, seed 20261110), sensitivity half-range 0.000187, **`Drag_Std` 0.000244 1/m**; residual RMS 5.42 mm
  (4.00–8.90), max 16.8 mm; C_d ≈ 0.66 with the *assumed* 0.122 kg. Untrimmed 0.009052 (5.89 mm); vacuum 12.3 mm. **No
  further term**: a shared bias gains 0.013 mm; a per-flight k and a Magnus term forecast 6.5 and 102 mm worse than
  the joint k (16.5 mm). **Validation**: 6.62 mm (4.06–11.92) trimmed, 6.82 mm untrimmed, per session 4.94–10.03 mm.
- **Residual structure**: correlated over 50–80 ms; next to a dropout 9.4 against 5.6 mm; an offset per present run
  removes 72 % of the sum of squares (shuffled 46–50 %): a measurement-side sign. **Estimated floor** (the RMS after a
  per-axis cubic, a heuristic): 4.25 mm (identification), 5.43 mm (validation). **Largest values**: residual 55.9 mm,
  jump across a dropout 89.4 mm, step inside a run 29.5 mm.
- **Noise**: 0.61 mm median (std(Δ²p)/√6); a 3D noise of 1.24 mm. **Not white**: the first differences' lag-1
  autocorrelation −0.07…−0.16 (white −0.50), growing with the lag in seconds (y: 0.9 mm at 10 ms, about 2.5 mm by
  60–80 ms), a white floor of about 0.3 mm per axis: hence M4 (§7.5).

### 4.3 Robot, cameras, markers

| Fact | Value | Source |
|---|---|---|
| UR10e reach, payload, joint ranges | 1300 mm, 12.5 kg, ±360° | UR10e technical specification |
| Joint speeds | base, shoulder 120 °/s; elbow, wrists 180 °/s | same |
| TCP speed, repeatability | 4 m/s; ±0.05 mm (ISO 9283) | same |
| Accelerations | 800 °/s² joint, 2500 mm/s² linear: **assumed** (UR's recommendation) | UR support article |
| Control rate; thrower to robot | 500 Hz; ≈ 4–5 m | the author's presentations |
| Cameras | 12, Qualisys 5+ / 6+ / 7+ family (model not named), latency down to 4 ms | Qualisys product info |
| Markers | super-spherical 16 mm, 2.0 g | Qualisys marker sheet |

### 4.4 CAD and scene placement

| What | File | Size |
|---|---|---|
| Ball R 30 mm, 6 markers | `CAD/Ball/STL/Orange_Ball_R30_Spherical_Markers.stl` | 91.9 mm over the markers |
| Conical cup (two printed parts) | `CAD/Gripper/STEP/Assembly_Conical_Cylinder.stp` | 153 × 190 × 153 mm |
| Robot stand | `CAD/Stand/STL/Assembly_Robot_Stand.stl` (608 435 triangles) | 800 × 1478 × 609 mm |
| Camera with mount; rail | `CAD/Mocap/STEP/Camera.stp`; `Holder.stp` | 325 × 432 × 130 mm; 11 × 5 m |

- **The robot area** (the 2022 hemisphere): r = 1.0 m centred at z = 0.3607 m, turned −90° about z. **The camera
  positions are in no file**: the mounts, the rail at 3.0 m above (0, −3) m and the cameras' boxes are *assumed*.
- **The stand**: MuJoCo 3.16.0 reads an STL of at most 200 000 faces, so the visual mesh is decimated once (Blender
  4.2.8 LTS, 20 000 faces); the collision is four boxes holding every CAD vertex.
- **The cup**: flange at 0, mouth at 190 mm, outer r 76.5 mm; a flat bottom at 5 mm to r 40.75 mm, a cone to r 67.0 mm
  at 180 mm, a lip narrowing the opening to 62.5 mm. A ball on the bottom has its centre 155 mm below the mouth; at the
  lip it is free to 32.5 mm off the axis. 383.6 cm³: 0.476 kg at the *assumed* 1 240 kg/m³ (PLA printed solid).

### 4.5 Compute (2026-11-10, 2026-11-17)

`dataset` of 10 000 flights: 23 s, 122.8 MB, peak 0.51 GB (18 s with the bounds). Training on the CPU, 7 000 flights,
1 thread: 2.5 s per epoch (`QT_BASIS`); more threads give no speed; MPS no gain at batch 128 and not deterministic; two
trainings of one seed give the same `weights.npz` byte for byte; `torch.save` does not. **Truth on the real validation
flights**: at N = 20 the prefix lasts 0.095 / 0.100 / 0.214 s (p10 / p50 / p90) and the ranges [0, 0.1) … [1.2, 2.0] s
hold 45, 41, 35, 32, 33, 45, 17 flights; at N = 40 0.195 / 0.215 / 0.984 s and 42, 37, 32, 21, 32, 31, 2 (43 flights).
A sample within 5 ms of a fixed horizon exists on only 10–33 of 45: real flights are scored in horizon ranges.

### 4.6 Numerical predictors on the simulated validation partition (F7, 2026-11-16)

`Default`, `Seed_1`–`Seed_4` (1 500 validation flights each); median, per-seed range in brackets; N = 10 / 20 / 40:

| Predictor | Remainder RMS [mm] | p90 [mm] | Catch point [mm] |
|---|---|---|---|
| LS | 62.8 (61.7–64.3) / 40.5 (40.5–42.3) / 25.0 (25.0–26.5) | 124.3 / 87.6 / 56.1 | 75.1 / 48.8 / 28.1 |
| GLS_M4 | 57.2 (57.2–59.3) / 34.7 (34.7–35.7) / 20.5 (20.5–21.7) | 112.4 / 68.3 / 43.0 | 68.2 / 40.2 / 21.8 |

GLS_M4 is better on every seed and N and was selected. The floor of about 10–12 mm at 0.05 s is M4's per-run offset. The
error scales linearly with `Observation.Noise.Scale` (GLS_M4 17.4 / 34.7 / 52.1 mm at N = 20 for 0.5 / 1.0 / 1.5).

### 4.7 The numerical predictors on the 45 real validation flights (F7, 2026-11-15)

The test ledger's rows 1 (GLS_M4) and 2 (LS), reproduced byte for byte by rows 3 and 4 (§4.9). Median / p90 remainder
RMS at N = 10 / 20 / 40: GLS_M4 52.1 / 120.7, 37.5 / 72.9, 20.6 / 42.1 mm; LS 54.0 / 138.8, 33.4 / 68.7, 20.7 / 41.4
mm. On real flights LS is level with GLS_M4: GLS_M4's simulated advantage rests on M4's error structure. Coverage with
the targets of that time 34 of 45 (42 of 45 with `Throw.Target_Area`).

### 4.8 The learned predictors on the simulated validation partition (F8, 2026-11-17)

Five seeds on the CPU, `Default`'s 1 500 validation flights; the median over the seeds of the per-seed median [mm]:

| Predictor | N = 10 | N = 20 | N = 40 |
|---|---|---|---|
| LS / GLS_M4 | 62.8 / 57.2 | 40.5 / 34.7 | 25.0 / 20.5 |
| `QT_BASIS` | 57.7 [57.2–58.2] | 35.3 [35.0–35.6] | 21.5 [21.4–21.9] |
| `QT_MLP` / `HYBRID` | 58.2 / 57.6 | 35.6 / 35.2 | 21.7 / 21.8 |

- `QT_BASIS` − GLS_M4, paired (bootstrap 95 %, 10 000 resamples, seed 20261117): −0.07 [−0.42, +0.36] / +0.18 [−0.05,
  +0.41] / +0.63 [+0.34, +0.80] mm; against LS −4.17 / −5.33 / −2.36 mm. Every learned predictor is level with GLS_M4
  at N = 10 and 20 and 2–5 mm better than LS; none is better than GLS_M4. The selection rule (fixed before the runs):
  overlapping medians, so the inference time decides (0.55 against 2.63 ms): **`QT_BASIS`**. The velocity's direction
  at the catch plane is within 0.2–0.5° (median) for every predictor.
- **Noise scale**: on error-free input LS and GLS_M4 are exact; `QT_BASIS` trained at 1.0 keeps 6.4 / 2.8 / 2.4 mm,
  trained on the five-scale mix (0, 0.25, 0.5, 1.0, 1.5) 1.3 / 0.8 / 1.0 mm, unchanged at 1.0.
- **The final predictor** (`QT_BASIS`, the mix, 16 epochs, patience 16, the position loss): 57.6 / 35.3 / 21.8 mm,
  against GLS_M4 −0.11 [−0.30, +0.22] / +0.19 [+0.00, +0.39] / +0.54 [+0.24, +0.74] mm, own error 1.34 / 0.78 / 0.92
  mm, trained in 1 010 s (5 seeds, 1 thread, idle). Two tuning trials gained at most 0.14 mm: not adopted.
- **Learning curve** (N = 20): 1 000 / 2 000 / 4 000 / 7 000 / 21 000 flights 37.8 / 35.8 / 35.4 / 35.3 / 35.1 mm. **No
  gain** from knowing the dropout patterns (pools A, B: within 0.4 mm) or from other dataset seeds (within 0.44 mm).
- **Inference per flight** (one process, 1 thread, idle, on mains; median / p90 [ms] at N = 10, 20, 40, with the
  queries at the remaining samples): `QT_BASIS` 0.42 / 0.45, 0.55 / 0.59, 0.84 / 0.87; `QT_MLP` 2.64 / 4.45, 2.63 /
  4.47, 2.65 / 4.45; `HYBRID` 15.73 / 17.01, 14.86 / 16.30, 13.48 / 15.46; LS 24.64 / 31.56, 32.80 / 50.42, 49.37 /
  90.22; GLS_M4 24.74 / 32.75, 35.17 / 52.30, 51.95 / 165.08. Against the re-planning period (10 / 5 / 3.33 ms at
  100 / 200 / 300 Hz) only `QT_BASIS` answers in time; the numerical ones spend about 15 ms integrating to the floor.

### 4.9 The final evaluation (F14, 2026-11-20)

Nine counted runs, the test ledger's rows 3–11, in an order and with a reading fixed before the data, each alone after
an idle check: GLS_M4 (3), LS (4), **the final predictor `QT_BASIS`** (5), `QT_BASIS` on `Seed_1`, `Seed_2` (6, 7), the
hybrid (8), `QT_BASIS` at one noise scale 0.5 / 1.0 / 1.5 (9–11). Real flights: 40 ms trimmed, robot frame by the
assumed transform, scored against the recorded samples; no prediction refused (0 of 4 921). A learned predictor's value
is the median over its 5 seeds; paired differences per flight with a bootstrap 95 % interval (seed 20261118).

| Real flights | N = 10 | N = 20 | N = 40 | End mismatch [ms] | Inference median / p90 [ms] |
|---|---|---|---|---|---|
| GLS_M4 | 52.1 / 120.7 | 37.5 / 72.9 | 20.6 / 42.1 | 20.8 / 18.4 / 16.6 | 24.50 / 30.30 … 49.51 / 171.00 |
| LS | 54.0 / 138.8 | 33.4 / 68.7 | 20.7 / 41.4 | 21.6 / 19.5 / 16.6 | 23.52 / 29.40 … 46.63 / 90.79 |
| **`QT_BASIS`** | 61.9 / 119.1 | 38.0 / 73.6 | 22.6 / 41.6 | 26.3 / 19.5 / 16.8 | 0.42 / 0.44 … 0.83 / 0.86 |
| `HYBRID` | 60.1 / 120.0 | 37.7 / 74.9 | 22.6 / 41.8 | 25.8 / 19.0 / 16.5 | 15.36 / 16.51 … 13.31 / 14.81 |

(median / p90 remainder RMS [mm]; `QT_BASIS`'s seed ranges of the median 53.5–63.3, 37.5–38.2, 22.1–22.7 mm.) Per
horizon range at N = 20 ([0, 0.1) … [1.2, 2.0] s), the median of the per-flight medians: GLS_M4 5.4, 10.8, 16.7, 25.8,
37.2, 53.8, 67.7 mm; `QT_BASIS` 5.5, 10.4, 16.4, 24.9, 38.2, 54.6, 64.6 mm.

**Paired on the real flights** [mm]: `QT_BASIS` − GLS_M4 −0.29 [−0.72, +0.26] / +0.45 [−0.71, +1.08] / −0.23 [−0.94,
+1.30]; `QT_BASIS` − LS −5.04 [−7.86, −1.36] / −0.03 [−3.49, +5.12] / −0.89 [−2.15, +0.66]. At N = 10 the 45 values
have a gap at their middle, so one flight's rank moves the median by 10 mm: the paired differences are the comparison.
**The reading, fixed before the data**: no interval against the better numerical predictor lies entirely above 0 and
no median is above 5.4 mm: **large sim-to-real gap: no, at N = 10, 20 and 40.** The final predictor is level with GLS_M4
at every N and answers in 0.4–0.9 ms against 24–50 ms: about 58× faster (58.4, the geometric mean over N = 10, 20, 40
and the simulated test and real sets of GLS_M4's median inference time over QT_BASIS's, from compare's stored tables;
§7.14).

**Sim-to-real** (real median against the simulated test partition's, `Default`): `QT_BASIS` 61.9 / 59.7, 38.0 / 36.3,
22.6 / 22.4 mm, ratio 1.04, 1.05, 1.01, the differences +2.2 [−22.6, +25.0], +1.7 [−3.6, +13.3], +0.1 [−7.9, +4.4]
mm; GLS_M4 ratio 0.88, 1.05, 0.97; LS 0.86, 0.80, 0.79. The shortest range on real flights 4.5–6.6 mm, at the level of
the 5.43 mm floor. **Support**: 42 of 45 validation flights inside; outside 200 Hz OFF Experiments 30 and 45 and
200 Hz ON Experiment 3; the reading holds inside. The noise-scale runs and the dataset seeds: every paired interval
against the final predictor spans 0 or lies within 0.73 mm.

**The simulated test partition** (`Default`; N = 10 / 20 / 40; no refusal, no crossing disagreement):

| Predictor | Remainder RMS [mm] | At the end [mm] | Catch point [mm] | Time to catch, p90 of abs [ms] |
|---|---|---|---|---|
| LS | 63.1 / 41.9 / 26.3 | 94.2 / 63.2 / 36.3 | 76.4 / 51.4 / 31.4 | 20.2 / 13.1 / 8.0 |
| GLS_M4 | 59.2 / 35.9 / 21.1 | 86.9 / 51.2 / 27.5 | 70.9 / 41.5 / 23.6 | 19.0 / 11.7 / 6.7 |
| **`QT_BASIS`** | 59.7 / 36.3 / 22.4 | 88.3 / 52.3 / 29.2 | 71.2 / 42.1 / 24.4 | 19.0 / 11.6 / 6.8 |

`QT_BASIS` − GLS_M4 paired +0.23 [+0.00, +0.46] / +0.16 [−0.02, +0.34] / +0.73 [+0.45, +1.00] mm; own error 1.3 / 0.8 /
0.9 mm. **The final predictor's weights** (SHA-256 of `weights.npz`; `train` on the CPU reproduces them byte for byte):
seed 0 `a312f42d7eddaa21027d3e26dc4e0b98f44ae897516a61979b46fd261741144f`,
seed 1 `6dec72dd5227faf6174c4e2793f618d4fef134f988269256a3967c337f47daee`,
seed 2 `61200affe873af37128f0fa9a2277f8e746164a5ebf869888d43ce025970089d`,
seed 3 `16d43c4537745cb6d17b87987cfda8d81a4f69a0f8665c3c20e9997d9a3d9ab0`,
seed 4 `586e2a47fa9952bcc5d8f5e5082b4b1bd480cb7253198d4751bc2d0b5c5528f0`.

### 4.10 The workspace: where the robot may catch (F4, 2026-11-21)

`derive.py workspace` on `Default`'s 7 000 train throws. A point qualifies when a catch pose (§7.11) exists for all 25
directions of the arrival cone (elevation 44–68°, azimuth −25…27°) with the margins 10°, 0.05 m, 0.5 m/s (*assumed*):
25 781 of 128 844 lattice points (0.05 m), 3.22 m³; 56.4 % of the 2022 hemisphere. **The ready pose** `Robot.Ready_Pose`
(29.972673, −104.916569, −88.564625, −57.375802, −61.349913, 0)° and **the catch area** `Areas.Catch` are chosen
together: the largest qualifying cuboid reached from the pose within 0.7 s at 800 °/s², shrunk by one step: centre
(0.7, −0.175, 0.9) m, size 0.3 × 0.75 × 0.5 m, **0.1125 m³**, the cup inside. **9.7 %** of the train throws pass through
it (62.4 % pass the hemisphere). Rest to rest from the ready pose at 800 / 400 °/s²: 0.947 / 0.370 m³ reached in 0.7 s;
catchable anywhere 39.9 / 28.2 %; **96.5 / 95.1 % of the throws through the area catchable in it**.

### 4.11 The catch decision on simulated throws (F9, 2026-11-21)

q 0.95, stages 20 → 40, margins from the validation partitions of `Catch_In` and `Catch_Near`, the truth over
`Areas.Catch` grown by 0.05 m; the decision ledger's rows 1 (`QT_BASIS`) and 2 (GLS_M4). No prediction refused.

| Dataset | °/s² | Not catchable | False-go, every `QT_BASIS` seed | False-no-go `QT_BASIS`, GLS_M4 |
|---|---|---|---|---|
| `Catch_In` | 800 / 400 | 125 / 142 | 0 [0, 2.91 %] / 0 [0, 2.56 %] | 27.9–28.1, 27.4 / 32.6–32.9, 32.1 % |
| `Catch_Near` | 800 / 400 | 4 681 / 4 718 | **0 [0, 0.079 %] / 0 [0, 0.078 %]** | 58.8–59.1, 58.6 / 61.6–61.8, 61.4 % |
| `Default` | 800 / 400 | 1 327 / 1 329 | 0 [0, 0.28 %] | 63.6–64.2, 64.2 / 67.3, 66.7 % |

(4 500, 9 000 and 1 500 throws.) **The reading**: ≤ 0.1 % for each seed: **met**; GLS_M4 had no false go either. The
20th sample comes 0.063 / 0.100 / 0.230 s after the release (p10 / p50 / p90); the ball enters the area 0.73 / 0.97 /
1.14 s (p5 / p50 / p95) later.

### 4.12 The interception on simulated throws (F10, 2026-11-21/22)

`QT_BASIS`, 5 seeds; the ladder N = 20 … 150 from the train partitions; the interception ledger's row 1 (3 746 s).

| Dataset | °/s² | False-go (upper) | False-no-go | F9 baseline | Intercepted of catchable | Of goes |
|---|---|---|---|---|---|---|
| `Catch_In` | 800 | 0 (2.91 %) | 11.3–11.8 % | 28.0–28.8 % | 85.0–85.9 % | 96.3–96.8 % |
| `Catch_In` | 400 | 0 (2.56 %) | 18.5–19.3 % | 32.7–33.6 % | 77.5–78.2 % | 95.5–96.0 % |
| **`Catch_Near`** | 800 | **0 (0.079 %)** | 45.1–45.5 % | 59.0–59.5 % | 53.1–53.3 % | 96.9–97.3 % |
| **`Catch_Near`** | 400 | **0 (0.078 %)** | 51.3–51.6 % | 61.7–62.1 % | 46.7–46.9 % | 96.4–96.7 % |
| `Default` | 800 | 0 (0.28 %) | 45.1–46.2 % | 63.6–65.3 % | 50.9–53.2 % | 93.7–97.9 % |
| `Default` | 400 | 0 (0.28 %) | 52.6–53.8 % | 67.3–67.8 % | 43.3–45.0 % | 91.4–96.2 % |

**R1 and R2 met** (60 102 goes). Along the paths: clearance ≥ 0.1152 m to the environment, ≥ 0.0575 m between the
robot's parts; time margin ≥ 0.025 s; lateral miss 10.2–13.2 mm (median), 28–34 mm (p95); tracking 0.067 mm (median).

### 4.13 The ball in MuJoCo against the reference integrator (F5, 2026-11-22)

The train partitions only: MuJoCo's ball against `Flight.Core.Simulate` (RK4, 1/600 s) at the scene's 0.002 s steps:
position p50 / p99 / max 0.70 / 0.88 / 0.98 µm (`Default`, 7 000), 0.71 / 0.89 / 0.94 (`Catch_In`), 0.71 / 0.88 / 0.95
(`Catch_Near`); velocity at most 1.39 µm/s; at the catch plane at most 0.79 µm and 0.13 µs. The scene's semi-implicit
step puts a ball placed with its velocity ahead by g·h·t/2 (12.4 mm median); placed half a step back it is leapfrog.
The options (200 throws, median / max): the force with the velocity as given 12.84 / 15.13 mm; with the half-step start
(chosen) 0.001 / 0.001 mm; MuJoCo's ellipsoid fluid model 11.88 / 13.72 mm (exact only under RK4, at which the held
robot drifts); the inertia model's drag per axis 36.8 / 83.2 mm. The bound (0.1 mm, 1 mm/s) is a third of the white
noise and under 2 % of F3's residual; the chosen option meets it about 100× over. 20.5 µs per step. **Early contacts**
(300 throws, the robot at q = 0): the floor first in 205, the stand in 57, the robot in 35, the cup in 3.

### 4.14 The physical catch: the contact, the cliff and the criteria (F11 Part A, 2026-11-22)

Scratch, simulated validation partitions only (`QT_BASIS` seed 0). **The contact needs a smaller step**: at 0.002 s
MuJoCo's default contact returns 0.05 at 7 m/s with 48 mm of penetration; a time constant below two steps is raised by
MuJoCo, so it is refused. **The nominal contact** (0.00025 s; time constant 0.004 s, damping ratio 0.7; the mean over 8
contact phases): restitution 0.194 at 7 m/s, 0.185 at 3, 0.162 at 1 m/s; 8.4 mm of penetration at 7 m/s; a single
bounce scatters by ±0.012. **The levels**: 0.12 (0.00157 s, 0.9) 0.135; 0.29 (0.004 s, 0.5) 0.284; 0.45 (0.3) 0.438.
**The cliff** (retained of the goes; `Catch_In` 800 / 400, `Catch_Near` 800 / 400 °/s²): 0.12 99.4 / 99.4 / 99.7 / 99.7
%; **0.20 97.4 / 97.1 / 98.5 / 98.3 %**; 0.29 25.3 / 26.1 / 30.8 / 28.7 %; 0.45 2.3 / 1.7 / 1.5 / 2.0 %. **The
criteria**: without rolling friction 96.0 % are retained but 16.3 % at rest, so *retained* is the criterion. At half the
cup's mass 95.8 % against 95.4 %: 0.476 kg stays *assumed*; at the cliff nothing is converged (each nudge moves it 10–20
points). **The recordings' floor bounce** gives 0.72 (0.69–0.75): the lab floor was covered with spring mattresses, so
it is not a value for the cup, and the recorded flights land about 30 mm above the floor datum.

### 4.15 The physical catch on simulated throws (F11, 2026-11-22)

The shipped scene, the nominal contact (*assumed* 0.20), `QT_BASIS`, 5 seeds (range, mean in brackets). Validation:
94 336 simulated goes; test, the counted read (catch ledger row 1): 60 102. **Every share holds for the *assumed*
contact.**

| Partition, dataset | °/s² | Retained, of the goes | Of the catchable (means) | F10 intercepted |
|---|---|---|---|---|
| validation `Catch_In` | 800 / 400 | 95.7–97.9 (97.2) / 94.8–97.8 (96.6) % | 84.6 / 76.9 % | 83.3–83.9 / 75.6–76.2 % |
| validation `Catch_Near` | 800 / 400 | 95.4–98.1 / 94.4–97.0 % | 52.8 / 47.1 % | 52.1–53.0 / 46.4–47.1 % |
| test `Catch_In` | 800 / 400 | 95.7–97.5 (97.0) / 95.0–97.4 (96.6) % | 85.7 / 78.5 % | 85.0–85.9 / 77.5–78.2 % |
| test `Catch_Near` | 800 / 400 | 95.6–97.5 / 95.3–97.6 % | 53.1 / 47.0 % | 53.1–53.3 / 46.7–46.9 % |

**Classes** (test, the 5 seeds summed, `Catch_In` 800 / 400 °/s²): caught 18 746 / 17 100, bounced out 476 / 490, rim
111 / 105, no go catchable 2 542 / 4 095, no go 625 / 710; no throw hit the robot or the stand, missed or was late; no
false go. **Readings**: R1 (the interception's rows equal F10's, 150 000) and R2 (one class per row, no robot contact
but the ball) hold; R3 descriptive. **The levels** (validation, seed 0, `Catch_In` 800 °/s²): 0.12 99.5 %, 0.20 97.9 %,
0.29 28.2 %, 0.45 2.1 %. With the true trajectory every go is retained; the offset at the mouth plane decides (0–15 mm:
41 396 of 41 397). Seed 4 retains fewer (95.4 % against 97.0–97.5 % on test). **The sweep** (validation, 1 500 throws,
the truth plus an error sized from the stored real-flight errors; `Catch_In` 800 / 400 °/s², of the catchable):
`QT_BASIS` 84.5 / 76.0 %; the real level 80.5 / 72.3 %; twice 69.5 / 59.9 %; the simulated level 80.4 / 72.7 %. At the
real level two throws go falsely (`Catch_Near` 3153, `Default` 8442), recorded, nothing adjusted.

### 4.16 The moving catch on simulated throws (F15, 2026-11-23/24)

`Catch_Mode` `moving`, the scales 1 / 0.75 / 0.5 / 0.25, `Stroke_Compute_Budget` 0.10 s. Validation 13 729 s (188 440
goes, levels and markers included); test, the counted read (catch ledger row 2), 8 099 s (60 102 goes). **Every share
holds for the *assumed* contact and the *assumed* compute budget.** Retained of the goes, static → moving:

| Partition, dataset | °/s² | Static | Moving | Of the catchable (means) |
|---|---|---|---|---|
| validation `Catch_In` | 800 / 400 | 95.7–97.9 / 94.8–97.8 % | 99.1–99.2 / 98.0–98.5 % | 84.6→86.3 / 76.9→78.1 % |
| validation `Catch_Near` | 800 / 400 | 95.4–98.1 / 94.4–97.0 % | 99.2–99.4 / 98.5–98.7 % | 52.8→54.1 / 47.1→48.4 % |
| test `Catch_In` | 800 / 400 | 95.7–97.5 / 95.0–97.4 % | 99.3 / 98.4–98.6 % | 85.7→87.7 / 78.5→80.0 % |
| test `Catch_Near` | 800 / 400 | 95.6–97.5 / 95.3–97.6 % | 99.5 / 98.5–99.0 % | 53.1→54.4 / 47.0→47.9 % |

**On test**: R1 holds (the same 60 102 goes); R2 holds (60 075 strokes; 27 goes planned static); the catch's R2 holds.
**R4 (a) met**, pooled: `Catch_In` 96.96 % [96.71, 97.20] → **99.28 %** [99.15, 99.40] at 800 °/s² (19 333 goes),
96.64 % → **98.51 %** [98.32, 98.69] at 400; `Catch_Near` 97.05 → 99.50 %, 96.78 → 98.70 %. **R4 (b) met**: the lowest
cell 98.4 % [97.9, 98.8]. **Not a pure gain**: the moving catch loses 80 / 191 goes the static one retains and retains
528 / 523 it loses; the 1 571 bounced-out balls become 7, but the rim takes 600 against 318. **The strokes** (test,
`Catch_In`, 800 / 400 °/s²): full scale 90 / 14 %; relative speed at the first cup contact 5.75 / 6.24 m/s (median);
start 0.195 / 0.202 s before the catch time; lateral miss 13.7 / 16.6 mm against static 10.7 / 11.3 mm. **The cliff**
(validation, seed 0, `Catch_In` 800 / 400): 0.29 98.6 / 95.6 %, 0.35 94.9 / 66.7 %, 0.45 68.1 / 15.4 % (with rigid
markers 96.4 / 90.3 %), 0.56 7.5 / 6.1 %: it moves from 0.20–0.29 to about 0.48 / 0.38. **The planner's own time**
(the shipped code, one process, one thread, 120 throws): moving 67 / 65 ms median, 117 / 152 ms p95, 14.5 / 23.3 % of
the passes over 0.10 s; static 8 / 11 ms, 57 / 58 ms, 8.1 / 9.2 % over 0.05 s. The simulation fixes a plan's time at
its budget (never the wall clock); both budgets stay assumptions.

## 5. Platform

- **macOS arm64**, verified; Linux unverified. **Python 3.11** at `/opt/homebrew/bin/python3.11` (`brew install
  python@3.11`). **MuJoCo 3.16.0**, pinned: if something needed does not work in it, the version is not changed.
- Pinned in `requirements.txt`: PyYAML, pytest, pytest-xdist 3.8.0 (test-only), numpy 2.4.6, torch 2.15.0 (only `train`
  imports it), mujoco 3.16.0, matplotlib 3.11.2, SciencePlots 2.2.2. Licences: numpy BSD-3, torch BSD-3-Clause (the
  wheel a compound of what it bundles), mujoco Apache-2.0, matplotlib PSF-based, SciencePlots, PyYAML, pytest MIT; their
  dependencies BSD, MIT, Apache-2.0, PSF-2.0 or MIT-CMU.
- The viewer is MuJoCo's passive viewer, on macOS under `.venv/bin/mjpython`; frames are written with the standard
  library. The figures use LaTeX when `latex` and `dvipng` are found; it is not a requirement.
- **The vendored robot**: MuJoCo Menagerie `universal_robots_ur10e` at commit `fb36886b`, BSD-3-Clause, "Copyright 2018
  ROS Industrial Consortium"; servos kp 5000, kv 500; never edited; 26 files, tree id
  `466f0ad824356ae94da5f514e55d630ca67071c7`.

## 6. Repository structure

```
README.md, LICENSE, CHANGELOG.md, CITATION.cff   the public page; MIT; the release history; how to cite
install.sh, requirements.txt, pytest.ini         macOS arm64 (§11); pinned; the test runner's configuration
Config/            ball_catching.yaml (§8.1), dataset.yaml and Datasets/ (§8.2), training.yaml (§8.3)
Data/              Recordings/ (the 90 recordings, §9.2), Legacy/ (§9.3), Generated/<Kind>/ (git-ignored)
CAD/               the CAD sources (§4.4)
models/            scene.xml, cup.xml, meshes/, scene_fingerprint.json, build.json, SHA256SUMS,
                   the vendored universal_robots_ur10e/ (§9.10)
scripts/           catch.py (dataset, train, evaluate, compare, catch, view), plots.py (§9.16),
                   Maintenance/ (derive.py: recordings, identify, observation, workspace, scene, dataset-check,
                   sweep; legacy_baseline.py; build_scene.py)
src/<Domain>/      §7
test/              mirroring src/; scripts/, Install/, Docs/, Legacy/ (the 2022 reference)
Media/             the README images: readme_<role>_<theme>.png, copies of plots.py's key set (§9.16); readme_hero.jpg,
                   readme_catch.gif, readme_prediction.gif from the video's build outside the repository (the
                   validation throw Catch_In 4748, seed 0, 800 °/s²; GIFs 720 px, 128 colours, 15 fps, ≤ 400 kB)
docs/              SPEC.md, KNOWN_ISSUES.md
```

`scripts/catch.py` holds the user's commands; `scripts/Maintenance/derive.py` what the shipped configuration and the
catch area were derived with. Every parser refuses an abbreviated option. **The earlier names** used in §4:

| Earlier, `scripts/catch.py ...` | Now |
|---|---|
| `validate --run DIR`; `test --run DIR` | `scripts/catch.py evaluate --run DIR [--partition test]` |
| `decide`, `intercept`, `catch` | `scripts/catch.py catch --run DIR [--partition test]` |
| `catch --sweep EVALUATIONS` | `scripts/Maintenance/derive.py sweep --run DIR --evaluations EVALUATIONS` |
| `dataset --check DIR` | `scripts/Maintenance/derive.py dataset-check DIR` |
| `recordings`, `identify`, `observation`, `workspace`, `scene` | `scripts/Maintenance/derive.py <the same name>` |

`src/` layers, lowest first: Configuration; Geometry; Flight; Recording, Observation; Dataset; Prediction; Learning;
Robot; Scene; Catch; Evaluation; View; Plot (the only one importing matplotlib). A layer imports only layers before
it; a domain's topic modules import its `Core.py`. Scripts reach `src/` through a path built from `__file__`.

## 7. Domains

### 7.1 Configuration
The only reader of §8's files: strict YAML (duplicate keys refused, only `true` / `false`), every problem collected and
named, an unknown key a WARNING; returns `*_Str` structures. A dataset's system copy is read for four sections (§8).

### 7.2 Geometry
Cuboid, sphere, hemisphere, quarter sphere; point inside, random point inside. **A point on a boundary is inside**, with
an allowance of 1e-12 m (`CONST_BOUNDARY_EPSILON`; a turned area moves a point by about 1e-16 m). Ported from 2022.

### 7.3 Recording
The only reader of §9.2 (`src/Recording/Core.py`). Sample j is on line j + 2; a step is reported at its later line.
- **Time**: t_k = t_{k−1} + n_k·T, r_k = ((ns_k − ns_{k−1}) mod 1e9)/(1e9·T), n_k = floor(r_k + 0.5). **A gap is a step
  of r_k ≥ 1.5.** A gap, a zero step and an irregular step (|r_k − n_k| > 0.05) are reported. **Ambiguous** (a hidden
  gap ≥ 1 s): a zero step, a jump over 10 m/s·Δt + 0.02 m, or `CONST_CITED_HIDDEN_GAPS`; each starts a piece of time.
- **Invalid input**: every problem collected with file and line, nothing written.
- **Segmentation** (`Segmentation.py`): around the highest sample (≤ 1.0 s each side, cut where z rises 0.1 m above its
  running minimum), a constant-acceleration fit trimmed while an end is off by more than τ = 20 mm, then grown back.
  Accepted when a_z ∈ [−10.5, −9.0] m/s², horizontal ≤ 1.5 m/s², the apex inside, ≥ 0.3 s, not cut short. 90 of 90; one
  flight flagged `Adjacent_Ambiguous`.
- **Split** (`Assign_Sets`): per folder by the SHA-256 of `"20261109/<folder>/<file name>"`, the first half
  identification: 45 / 45, from names and the seed only.
- **Identification** (`Identification.py`, F3): one k; each flight's initial state by Gauss-Newton
  (`Flight.Core.Fit_Initial_States`); k by profiling; a non-identification flight refused by name; the first 40 ms
  (`CONST_IDENTIFICATION_TRIM`) dropped inside it only; `Drag_Std` the bootstrap sd and the half-range in quadrature.
- **Transform to the robot base — assumed, configurable**: p_robot = Rz(180°)·p_QTM + (0.827, −4.670, 0.000) m, mapping
  the flights' median descent through z = 0.8607 m, (0.827, −4.170) m in QTM, onto (0, −0.5, 0.8607) m in front of the
  robot; a convention, not a fit. Being rigid, it leaves a prediction error at a given time unchanged.

### 7.4 Flight
`Flight_Model_Str` (`Gravity`, `Drag` k, `Drag_Std`), a = g − k·|v|·v (`Drag` = 0 the 2022 vacuum path bit for bit);
midpoint, Heun, RK2, RK4 of 2022; `Simulate(x_0, t, *, Flight_Model_Str)`: RK4 on 1/600 s (`CONST_SIMULATION_STEP`,
converged to 1.2e-13 m) and one partial step per time, so a time's state does not depend on the other times;
`Simulate_At`. The model is keyword-only, never defaulted. Ball: 60 mm, mass 0.122 kg (0.110 + 6 × 0.002), *assumed*.

### 7.5 Observation
Turns a simulated flight into samples at capture times; measured on the identification flights only; reads no file.
- **Donors**: the identification flights' sample times from the 40 ms trim; pools A / B by the SHA-256 of
  `"20261111/<name>"`. **Replay**: a donor sample stands for its frame period; frame 0 always present. **Patterns**:
  `composed` (default) cuts at exponential intervals of mean 0.25 s (*assumed*), replaying one of the 10 donors nearest
  in duration per segment at the same time since the release (presence 0.88 in the first 0.1 s, 0.19–0.22 at 0.5–0.8
  s); `recorded`; `none`. **Rates**: 100 / 200 / 300 Hz replay their own donors; **370 Hz the 300 Hz ones, *assumed***.
- **Error M4**, per axis, times `Noise.Scale`: white Student-t; a drift (Ornstein-Uhlenbeck in time); an offset constant
  within a run and after a dropout of lost time L b·o + √(1 − b²)·s·η, b = exp(−L/`Offset_Time`). **Bounds**: offset
  norm ≤ 0.070 m, white norm ≤ 0.040 m (55.9 and 29.5 mm × 1.25 *assumed*, rounded up), beyond drawn again.
- **Estimator**, deterministic: within a run E[½Δ²] = s_w² + s_c²(1 − e^(−τ/T_c)) for τ ≤ 0.1 s; across a dropout plus
  s_o²(1 − e^(−L/T_o)); non-negative least squares; dof by the tail ratio; pairs sorted canonically.
- **`Observation.*` are effective calibration values**, not physical quantities: 10 of 19 error statistics lie outside
  their bounds; the simulated observations are rougher than the real ones (7.16 against 5.42 mm), the conservative
  direction. Bounds are 95 % bootstrap intervals over the identification flights, never a tolerance chosen to pass.

### 7.6 Dataset
Flight i draws from stream i of `np.random.SeedSequence(Dataset.Seed).spawn`: the rate, candidates until one is
accepted, the pattern, the error. **Release**: a Gaussian, drawn again until in `Areas.Throw`. **Target**: uniform in
`Throw.Target_Area` (the half of an upper hemisphere of r 1.5 m facing the thrower: 42 of 45 identification flights
in the support) or in `Areas.Catch` grown by a margin. **Flight time** from an apex h: t = (v_z + √(v_z² − 2g(z_t −
z_0)))/g, v_z = √(2g(h − z_0)). **Release velocity** (`Shoot`): Newton until the target is missed by < 1e-9 m.
**Truth**: `Simulate` at every frame to the floor time (z = 0.030 m). **Observations**: at j / rate exactly.
**Partitions**: largest-remainder sizes, index blocks train, validation, test; a run refuses a dataset of another hash.

### 7.7 Prediction
A predictor returns a `Trajectory_Str` (§2.13.13). **`GLS_M4`** fits the state at the first sample by Gauss-Newton
under F3's model, weighted by M4's covariance between the samples (`Observation.Core.Error_Covariance`: white, drift
and offset with each dropout's lost time; the estimate of a Kalman smoother carrying M4's states); it integrates on the
1/600 s grid to the first descent through 0.030 m and answers by cubic Hermite interpolation (4e-14 m from
`Simulate`); a time outside [t_0, t_end] raises `Prediction_Error_Cls`. `LS` is refused by name.

### 7.8 Learning
**`QT_BASIS`**, trained in PyTorch, run in NumPy (`Predictor.py`); only `train` imports PyTorch. A GRU encodes the
prefix (`Model.Hidden`); the decoder gives a Chebyshev series per axis over `CONST_HORIZON` = 2.0 s, degree
`CONST_BASIS_DEGREE` = 6 (own error 0.20 mm; 0.97 mm at 5). **Inputs**: each sample at its own time stamp relative to
the last (time, position, finite-difference velocity, frames lost, step), no absolute position; one model for every N.
**Velocity**: the derivative series. **Contract**: the end is the root of the height at 0.030 m (Newton to 1e-13 s);
a prefix below the floor or not landing within 2.0 s is refused; each query is its own matrix product of a fixed
shape, so it is the same bit for bit alone, in a subset or reversed. **Loss**: the RMS position error over the
remainder at 48 times per prefix; Adam, one cycle; the best epoch kept. **The reference device is the CPU**, 1 thread,
deterministic: a seed is reproduced bit for bit; a run on `mps` is refused by the test command. `QT_MLP` and `HYBRID`
are refused by name.

### 7.9 Robot
`Find_Robot`: the six joints (ranges ±360°, the elbow ±180°), the flange and tool sites; a joint beyond its range is
refused, never clamped. **The closed-form IK** from the model's UR geometry (offset 0.174 m, joint 5's 0.12 m, the tool
0.29 m beyond the wrist): up to 8 solutions, matching a numerical IK within 1e-6 rad. `Move_Time` rest to rest.
`Tool_Jacobian`, the speed margin, `Line_Rates` (F15). **Clearance**: 41 environment pairs and 23 self pairs by
`mj_geomDistance`, never a parent and its child, never the ball; 11 constant pairs left out. **Motion** (`Motion.py`):
`Profile`, each joint's time-optimal second-order profile to rest (no synchronisation; equal to Ruckig 0.19.4 within
2.2e-15 s, checked outside the repository); `Stroke`; `Check_Profile`, poses no robot point moves more than
`Path_Step` apart, each with the clearance plus its steps (certified); `Track`, the vendored servos with a feed-forward
of MuJoCo's inverse dynamics.

### 7.10 Scene
`Build_Scene` (`Scene_Spec`, then `Compile_Scene`) reads `models/scene.xml`, attaches the cup to the flange, the robot
(prefix `ur10e/`) to `robot_mount` at z 0.180 m, draws the areas; the vendored file is read as it is. Tool centre point
`ur10e/cup_mouth`, 0.190 m from the flange. Colliding: the floor, the stand's four boxes, the ball's sphere (the markers
visual), the cup's cylinder (for the clearance), the cup's shape for the ball (a bottom, 24 wall staves, 24 lip boxes,
meeting the ball only) and the robot's nine. implicitfast, 0.002 s, gravity 9.81 m/s². Robot 32.700 kg, cup 0.476 kg,
ball 0.122 kg. A build MuJoCo warns about is refused; no `MUJOCO_LOG.TXT`. **The ball** (`Ball.py`, F5):
`Place_Ball` (the velocity v − h/2·a(v)), `Apply_Drag` (−m·k·|u|·u at u = qvel + h/2·a(qvel), before every step),
`Ball_State`, `Find_Markers` (six markers 37.97 mm out, r 7.92 mm); another gravity or a medium is refused.

### 7.11 Catch
**The catch pose** (`Core.py`): a joint vector qualifies when every joint is 10° inside its range, the clearances ≥ 0.05
m, the speed margin ≥ 0.5 m/s (*assumed*); **a point qualifies** when all 25 axes of the cone have one. `Workspace.py`:
the map, the shrink by one step, the reach rest to rest; **the ready pose and the catch area chosen together** (B800):
the pose maximising the shrunk largest cuboid reached within 0.7 s at 800 °/s² (heuristic search). 0.7 s: the ball
enters the 1.0 m sphere 0.76 s (5 %) / 0.99 s (median) after its 20th sample.
**The decision** (`Decision.py`, F9). *Truth*: the true trajectory every 10 ms in `Areas.Catch` grown by 0.05 m, each
sample's fastest catch pose for its own arrival direction; catchable at N when its reach ≤ its time − the N-th sample's
− 0.10 s (`CONST_ALLOWANCE`). *Margins*: the errors split along the true velocity (cross-track per axis, time early),
quantile 0.95 per stage and 0.05 s bin (< 50 errors: no go). *Decide*: go when a predicted sample lies in the area
shrunk by the margins and is reached in time; a refused prediction is no-go. Clopper–Pearson 95 % rates.
**The interception** (`Interception.py`, F10): from N = 20 every sample decided with the ladder's margins (train
partitions); one plan at a time, started at the newest sample (+ `Latency`), ready `Compute_Budget` later, its goal the
catch pose for that prediction; accepted when the robot rests `Guard` before the catch time on a certified path; no
abort. *Result*: the true ball's first crossing of the mouth plane (miss ≤ 32.5 mm, angle ≤ `Angle_Tolerance`, at rest).
`Intercept_Chunk` takes two keyword arguments (a throw's predictor, a plan callback) for the catch and the sweep.
*Moving* (F15): rates r of a unit cup speed along v at the catch pose, w_max = min_j v_j/|r_j|, per scale w = s·w_max;
the ball meets the bottom d/(|v| − w) after the catch time (d = 0.155 m); the stroke peaks then and brakes; the approach
rests at the stroke's start; the first scale in time and certified, else static; `Stroke_Compute_Budget` per plan; the
stroke is not re-planned once it runs; `at_rest`, `in_stroke` recorded.
**The physical catch** (`Physical.py`, F11): in memory the step 0.00025 s (8 per control tick), the ball's friction,
contact dimension 6, priority 1, its solref. The loop holds the robot, places the ball at the first tick after the
release, drives the servos and the drag, and ends at the catch time + `Settle_Time` or the floor. Read: the first
contact (cup bottom, wall, lip, robot, stand, floor), any robot contact but the ball (R2), *inside*, the relative speed,
*at rest*, *retained*. **Classes**: `no_go` (outside the workspace), `no_go_catchable` (missed), unsimulated; `caught`,
`late`, `hit_robot`, `hit_stand`, `bounced_out`, `rim`, `missed`. `Ball_In_Cup` is the one definition of *inside*. *The
markers variant* (moving, validation): six rigid marker spheres in memory, *assumed* rigid. **The sweep**
(`Robustness.py`): the truth plus an error at a scale of the stored real-flight errors, one quantile and direction per
throw (seed 20261122), kept through the re-plans: harsher than a predictor's.

### 7.12 Evaluation
- **Position error versus horizon** [mm]: simulated at `CONST_HORIZONS` {0.05, 0.1, 0.2, 0.3, 0.5, 0.8 s, the end},
  exact truth; both kinds in `CONST_HORIZON_RANGES` [0, 0.1), [0.1, 0.2), [0.2, 0.3), [0.3, 0.5), [0.5, 0.8), [0.8,
  1.2), [1.2, 2.0] s, one value per flight (the median), the count; fewer than 10 flights: listed, no statistics.
- **Remainder RMS** [mm] over the true trajectory from t_0 (past the predicted end the state held there); **end
  mismatch** [ms]; **catch-point**, **time-to-catch** errors and **crossing disagreement** at the plane z = 0.8607 m
  (`Catch_Height`, the robot area's centre + r/2; `both`, `disagree`, `passed`, `neither`) and **velocity errors**:
  simulated only; **inference time** [ms]. A refused prediction is its own outcome.
- **Comparison** (`Comparison.py`): stored rows only; a run's statistic the median over its seeds; paired differences
  per flight with a bootstrap 95 % interval (10 000 resamples, seed 20261118); sim-to-real against the first dataset's
  test partition; the real flights inside and outside the support.

### 7.13 View
**The record** (`Record_Throw`): the interception through `Intercept_Chunk` (every prediction with its wall time, each
plan), a go simulated by `Simulate_Catch` with the nominal contact; a no-go not simulated. The display's own
predictions (`--compare`, `--every`) never reach the decision or a plan. `Pick_Throw` refuses a test throw. **Shown**:
the samples, the newest prediction, *waiting* / *deciding* / *GO at N* / *NO-GO*, the plan, the class, the safety
sentence. **The look** follows a reference diagram: the zones in `#FF0000`, `#4576C4`, `#6CB145` (the flight zone drawn
from the throw area's face), the samples red up to N = 20 with "Throw Detection" (no detector runs), the prediction
(`QT_BASIS` `#70AD47`) with "Catch Prediction", the ball as its sphere. Cameras `side` (orthographic, default),
`overview`, `cup`, `thrower`. **The theme** (`Theme.py`, dark by default) is applied in memory: `models/` untouched, the
fingerprint changed in visual fields only. Frames up to 3840 × 2160 px; start-up 23 s (the area's map).

### 7.14 Plot
`Core.py`, the data: each figure's arrays from the stored files by their own readers, or computed by the catch's code.
The restitution is computed when drawn (`Mean_Rebound`): 0.133 / 0.192 / 0.281 / 0.347 / 0.436 / 0.560; a cliff is read
where the retained share falls below one half. The prediction's throw is chosen by a rule (`Choose_Throw`); the
prediction error from the final evaluations' `summary.csv` (median solid, 90th percentile dotted); the bars show the
median with the range over `QT_BASIS`'s 5 seeds and the speed-up as a geometric mean; the motion's two records equal
the stored rows byte for byte. A key variant (`Look_Str.Key`) may leave series or panels out: the README's real figure
is the 2D figure's y-z view and error. `Style.py`: SciencePlots' `science`, LaTeX when found, dark (`#0D1117`) and
light; each entity's colour at 4.5:1 or more (a test); "no go" the lighter grey, "no go, catchable" the darker (light
`#6E7781`, 4.55:1; `#57606A`, 6.39:1); PDF and PNG at 300 dpi. `Figures.py`: every data artist carries a gid.

## 8. Configuration

Three kinds of file, read by §7.1 through `--config`: the system (§8.1), the datasets (§8.2: `Config/dataset.yaml`
and six in `Config/Datasets/`, so a clean clone can generate every dataset named) and the training (§8.3). Every field
is REQUIRED. **Stored copies** (a dataset's two files, a run's training file) are records, never edited; a dataset's
system copy is read for `Areas`, `Transform`, `Flight` and `Observation` only (`CONST_DATASET_SECTIONS`).

### 8.1 System — `Config/ball_catching.yaml`

| Field | Meaning | Unit | Value and source |
|---|---|---|---|
| `Areas.Throw` | the releases' support | m | centre (−0.125, −5.25, 1.275), size (3.67, 2.42, 0.91) |
| `Areas.Flight` | the flight cuboid | m | (0, −3, 2.25), (4, 4, 3.5): the 2022 area at −y |
| `Areas.Catch` | where the robot may catch | m | (0.7, −0.175, 0.9), (0.3, 0.75, 0.5): `derive.py workspace` |
| `Areas.Robot` | the catch hemisphere | m, deg | (0, 0, 0.3607), (0, 0, −90), r 1.0 |
| `Transform.Orientation`, `.Position` | QTM to robot | deg, m | (0, 0, 180), (0.827, −4.670, 0.000), *assumed* |
| `Flight.Gravity`; `.Drag`, `.Drag_Std` | the flight model | m/s²; 1/m | 9.81 *assumed*; `derive.py identify` |
| `Observation.White_*`, `.Drift_*`, `.Offset_*` | M4 | m, s, - | `derive.py observation` |
| `Observation.Offset_Bound`, `.White_Bound` | the largest norms | m | 0.070, 0.040 |
| `Robot.Joint_Speed` | speed limits | deg/s | 120, 120, 180, 180, 180, 180 (datasheet) |
| `Robot.Joint_Accelerations` | the nominal, then the second | deg/s² | 800, 400, *assumed* |
| `Robot.Ready_Pose` | where the robot waits | deg | (29.972673, −104.916569, −88.564625, −57.375802, −61.349913, 0) |
| `Catch.Joint_Margin`, `.Clearance`, `.Speed_Margin` | the pose's margins | deg, m, m/s | 10, 0.05, 0.5, *assumed* |
| `Catch.Arrival_Elevation`, `.Arrival_Azimuth` | the cone | deg | [44, 68], [−25, 27] |
| `Catch.Workspace_Step` | the lattice step | m | 0.05 |
| `Workspace.Dataset` | the workspace's throws | path | `Data/Generated/Datasets/Default` |
| `Decision.Datasets` | the datasets `catch` runs, with hashes | path, hex | `Catch_In`, `Catch_Near`, `Default` |
| `Decision.Calibration` | whose validation partitions calibrate | path | `Catch_In`, `Catch_Near` |
| `Decision.Quantile`, `.Stages` | the margins | - | 0.95; [20, 40] |
| `Interception.Stages` | the ladder | - | [20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 120, 150] |
| `Interception.Calibration` | whose **train** partitions calibrate | path | `Catch_In`, `Catch_Near` |
| `Interception.Latency`, `.Guard` | timing | s | 0.004, 0.03, *assumed* |
| `Interception.Compute_Budget`, `.Stroke_Compute_Budget` | one plan, fixed | s | 0.05, 0.10 (§4.16: assumptions) |
| `Interception.Path_Step`, `.Angle_Tolerance` | path check; crossing | m, deg | 0.0025, 10 *assumed* |
| `Interception.Catch_Mode`, `.Stroke_Scales` | the catch | - | `moving`; [1.0, 0.75, 0.5, 0.25] |
| `Contact.Step`, `.Time_Constant`, `.Damping_Ratio` | the contact | s, s, - | 0.00025, 0.004, 0.7 (*assumed* 0.20) |
| `Contact.Friction` | sliding, torsional, rolling | -, m, m | 0.5, 0.005, 0.0001, *assumed* |
| `Contact.Settle_Time`, `.Rest_Speed`, `.Rest_Time` | the criteria | s, m/s, s | 2.5, 0.05, 0.2, *assumed* |
| `Contact.Levels` | sensitivity (time constant, ratio) | s, - | [0.00157, 0.9], [0.004, 0.5 / 0.4 / 0.3 / 0.2] |
| `Contact.Sweep_Scales`, `.Sweep_Throws` | the sweep | -, - | 0, 0.5, 1, 1.5, 2, 3; 1 500 |

The output paths (`Recording.*_File`, `Identification.*_File`, `Observation.Summary_File`, `Workspace.*_File`, the
`Output_Folder`s and `Contact.Ledger_File`) are repository-relative under `Data/Generated/`. Areas are in the robot
frame (§14.6) with the thrower at −y (§14.4).

### 8.2 Dataset — `Config/dataset.yaml` (F6) and `Config/Datasets/`

`dataset.yaml` is `Default`; `Config/Datasets/` holds `Noise_0`, `Noise_0_25`, `Noise_0_5`, `Noise_1_5` (`Default` at
another `Noise.Scale`) and `Catch_In`, `Catch_Near` (seeds 20261121, 20261122; partitions 0.10 / 0.45 / 0.45; margins 0
and 0.25 m; `Catch_Near` 20 000 flights).

| Field | Meaning | Unit | Value and source |
|---|---|---|---|
| `Dataset.Name`, `.Seed`, `.Flights` | folder, the streams' root, size | - | `Default`, 20261111, 10 000 |
| `Dataset.System_Configuration` | the system file read | path | `Config/ball_catching.yaml` |
| `Partitions.Train`, `.Validation`, `.Test` | ratios | - | 0.70 / 0.15 / 0.15, *assumed* |
| `Throw.Release_*`, `.Apex_*` | Gaussians of the release and apex | m | `observation`: 44 identification releases |
| `Throw.Flight_Time`, `.Release_Speed` | accepted ranges | s, m/s | [0.95, 1.40], [5.5, 8.0], *assumed* margins |
| `Throw.Target_Area` | the facing half of an upper hemisphere | m, deg | centre (0, 0, 0.3607), (0, 0, −90), r 1.5 |
| `Throw.Target_Catch_Area.Margin` | instead: `Areas.Catch` grown by it | m | 0 (`Catch_In`), 0.25 (`Catch_Near`) |
| `Observation.Rates`, `.Noise.Scale` | rates; the error's multiplier | Hz, - | [100, 200, 300]; 1.0 |
| `Observation.Dropout.Pattern`, `.Donors` | the pattern; the donor pool | - | `composed`; `all` |
| `Output.Folder` | the datasets' parent | path | `Data/Generated/Datasets` |

### 8.3 Training — `Config/training.yaml` (F7, F8)

| Field | Meaning | Unit | Value |
|---|---|---|---|
| `Run.Name`; `Model.Name` | the run's folder; the predictor | - | `QT_BASIS`; `QT_BASIS` or `GLS_M4` |
| `Run.Datasets[].Path`, `.Hash` | the datasets scored, each `flights.npz`'s SHA-256 | path, hex | the manifests |
| `Input.Observed_Samples` | the values of N | - | [10, 20, 40], *assumed* |
| `Model.Hidden`; `Run.Seeds`; `Run.Device` | the state; the seeds; the device | - | 128; [0, 1, 2, 3, 4]; `cpu` |
| `Training.Datasets[]` | trained on (train pooled, validation stops) | path, hex | the five noise scales of `Default` |
| `Training.Flights` | flights per dataset | - | 7 000 |
| `Loss.Velocity_Weight` | the velocity term's weight | s | 0 |
| `Optimiser.Learning_Rate`, `.Batch_Size`, `.Max_Epochs`, `.Weight_Decay` | Adam | - | 0.002, 128, 16, 0 *assumed* |
| `Early_Stopping.Patience`, `.Min_Delta` | stopping | epochs, m | 16 (= the epochs), 0.0001 |
| `Output.Folder` | the runs' parent | path | `Data/Generated/Models` |

The learning fields are REQUIRED for a learned predictor (a WARNING for a numerical one). A run's flight model and M4
are those of its datasets' stored system copies, which must agree with their manifests and with each other.

## 9. Data formats and units

### 9.1 Units
Length [m] (the CAD's mm converted in the scene's build); time [s] (the recordings' ns converted by their reader only);
angle [rad] (configuration fields in [deg] converted by the loader); mass [kg]. Invalid input of any reader: every
problem collected with its location, the run refused, nothing written. Readers take `--config FILE`, writers `--out`.

### 9.2 Recordings (input, committed, never changed)
`Data/Recordings/Frequency_<rate>_Hz_Active_Filtering_<ON|OFF>/Experiment_<n>.txt`, header `Time,Data_X,Data_Y,Data_Z`.
`Time` is `header.stamp.nsecs` of the ROS topic `/qualisys/Ball/pose` (`geometry_msgs/PoseStamped`), nanoseconds within
the second only; `Data_X..Z` the 6DOF body "Ball" in the QTM frame [m]. Recorded with the ROS package
`KTH-SML/motion_capture_system` and `record_data.py` (`rosrun mocap_qualisys record_data.py <samples> <name>`, a fixed
sample count per file); the rate set in QTM; "Active filtering" the QTM camera setting.

### 9.3 Legacy inputs and the baseline (committed)
`Data/Legacy/Experiment_1_<method>_DRAG_False.txt` (header `r_0_X,r_0_Y,r_0_Z,r_n_X,r_n_Y,r_n_Z,t_f`, 1000 rows, the
`t_f` defect of §4.1); `Data/Legacy/Baseline/<method>_<Last|Mean|Mean_Opt>.txt` (1000 errors [m], `%.10e`) and
`SHA256SUMS`. `scripts/Maintenance/legacy_baseline.py --python <dir>/bin/python` (a Python 3.10 with NumPy 1.22.4,
pandas 1.4.4, Matplotlib 3.5.3) regenerates and compares them (114 s, 12 of 12 identical). Read by
`test/Legacy/Reference.py` only; the slow test reproduces all 12 000 throws (54 s), the per-commit tier 50 per file.

### 9.4 Recording summary and steps (generated)
`derive.py recordings` writes the summary (per recording: `file`, `rate_hz`, `filtering`, `samples`, `wraps`, `gaps`,
`zero_steps`, `irregular_steps`, `ambiguous_steps`, `set`, `segmented`, `reason`, the flight's lines, duration,
coverage, fit, residuals, first and last sample in the robot frame) and the steps (`file`, `line`, `kind`, `reason`,
`ratio`, `frames`, `jump_m`). CSV, a header, LF, `%.6f`, no timestamp. Exit 0 written, 1 refused, 2 usage.

### 9.5 Dataset and manifest (generated)
`scripts/catch.py dataset` writes `Data/Generated/Datasets/<Name>/`: `flights.npz`, `manifest.json` and byte copies of
the two configurations; `derive.py dataset-check DIR` regenerates and compares. **`flights.npz`** (`Dataset.Core` only;
`np.savez` uncompressed, float64, `format_version` 1): per flight `partition` (0 / 1 / 2), `rate`, `release_state`,
`target`, `apex_drawn`, `flight_time`, `floor_time`, `draws`; ragged with offsets `truth_*`, `observed_offset`,
`observed_frame`, `observed_time`, `observed_position`, `donor_*`; robot frame. **`manifest.json`** (one writer for
datasets, runs and evaluations): sorted keys, no wall-clock time; the configuration's SHA-256 and copy, the seeds, the
data's SHA-256, the model hash, git commit and dirty flag, platform, versions. Written through a temporary folder; an
existing target refused. Exit 0 / 1 / 2.

### 9.6 Runs (generated)
`scripts/catch.py train` writes `Data/Generated/Models/<Run.Name>/`: the configuration's copy, `manifest.json` and per
seed `seed_<s>/` with `weights.npz` (`format_version` 1, the model's name, every weight in float32; the same weights
give the same bytes), `normalisation.json` and `epochs.csv` (`epoch`, `train_loss_m`, `validation_loss_m`,
`validation_position_m`). A partition is selected only by `Dataset.Core.Partition_Indices`, which the leakage tests
replace. A numerical run holds its configuration and manifest only.

### 9.7 Evaluations and the test ledger (generated)
`evaluate --run DIR [--partition validation|test]` writes `Data/Generated/Evaluations/<Run.Name>/<partition>/`:
`per_flight.csv` (`flight`, `partition`, `dataset`, `n`, `prefix_duration_s`, `rate_hz`, `remainder_rms_mm`,
`end_mismatch_ms`, `catch_point_error_mm`, `time_to_catch_error_ms`, `crossing`, `inference_ms`, `refusal`, `seed`;
the catch fields empty on real flights), `per_horizon.csv` (`flight`, `partition`, `dataset`, `n`, `horizon`, `truth`,
`position_error_mm`, `samples`, `direction_error_deg`, `speed_error_m_s`, `seed`), `summary.csv` (`partition`,
`dataset`, `seed`, `n`, `group`, `value`, `metric`, `count`, `mean`, `median`, `p90`, `max`), `predictions.npz`
(`format_version` 2), `coverage.csv` (test only) and `manifest.json`. `--partition test` first appends a row to
`test_ledger.csv` (`evaluation`, `run`, `model`, `configuration_sha256`, `dataset_sha256`, `commit`, `dirty`) and
refuses a dirty run, a run not trained on the CPU, a mismatched hash and an existing folder. `compare --evaluations DIR
[...] --name NAME [--against RUN ...]` reads only the evaluations' manifests, `per_flight.csv`, `per_horizon.csv` and
`coverage.csv` and writes `Data/Generated/Comparisons/<NAME>/`: `summary.csv`, `paired.csv` (`median_mm`, `low_mm`,
`high_mm`), `sim_to_real.csv`, `manifest.json`. CSV: a header, LF, `%.6f`. Exit 0 / 1 / 2.

### 9.8 Identification summary, flights and ledger (generated; F3)
`derive.py identify` (70 s) identifies k, evaluates the 45 validation flights and appends a ledger row (`evaluation`,
`commit`, `dirty`, `drag_per_m`, `drag_std_per_m`) before writing the summary (`kind` identified / sensitivity /
validation, the variant, the session, k and its sd terms, the median residuals and floor) and the flights (one row per
fitted flight with its initial state and residuals).

### 9.9 Observation summary (generated; F6)
`derive.py observation` reads the identification flights only (39 s, 0.8 GB): M4's estimate and spread, the largest
residuals and the bounds, the 19 error statistics and 6 diagnostics (each the median over 5 synthetic sets), the pattern
statistics per rate, the releases, the 12 throw-shape statistics, the coverage. Columns `section`, `statistic`, `rate`,
`real`, `low`, `high`, `generated`, `inside`, `generated_min`, `generated_max`.

### 9.10 The scene's files and its fingerprint (committed; F4)
`models/` is written by `scripts/Maintenance/build_scene.py --write` only; without it the script checks (exit 0
identical, 1 differs, 2 could not check). `scene.xml`, `cup.xml` (MJCF, m, rad); `meshes/` (binary STL, mm);
`SHA256SUMS` (the 26 vendored files); `build.json` (the inputs' and meshes' SHA-256, the stand's tool, the vendored
source); `scene_fingerprint.json` (`format_version` 1: the model's sizes, options and per element the fields that decide
what it computes and shows, nine significant digits; `environment` left out of a comparison). `derive.py scene` prints
`Fingerprint: identical to the stored one` (exit 0 / 1 / 2). A change of the scene is the build script, `--write` and
the fingerprint in their own commit.

### 9.11 The workspace's files (generated; F4)
`derive.py workspace` reads `Workspace.Dataset`'s train partition only, writes `workspace_map.npz` (`format_version` 1;
the points, the axes, `qualifies`, the failing axis and criterion, `poses` M×25×8×5 float32, `largest_cuboid`,
`catch_area`, `ready_pose`; 110 MB), `workspace_summary.csv` (quantity, value, unit) and `manifest.json`.

### 9.12 The decision's files and its ledger (generated; F9)
**One command writes §9.12–9.14**: `scripts/catch.py catch --run DIR [--partition validation|test] [--workers N]` runs
the decision, the interception and the physical catch and writes `Decisions/`, `Interceptions/`, `Catches/`
`<Run.Name>/<partition>/` (a moving catch `<Run.Name>/moving/<partition>/`); an existing folder of any stage is refused.
The decision: `per_throw.csv` (`dataset`, `flight`, `seed`, `acceleration_deg_s2`, `catchable`, `decision`, `go_stage`,
`outcome`, `refused`), `summary.csv` (the rates with Clopper–Pearson bounds, `go_by_stage`), `margins.csv` (validation:
`seed`, `n`, `bin`, `horizon_low_s`, `samples`, `cross_x_m`, `cross_y_m`, `cross_z_m`, `time_s`), `manifest.json` (the
test partition refused unless it matches the stored validation). `decision_ledger.csv`: a closed record of 2 rows.

### 9.13 The interception's files and its ledger (generated; F10)
The same bytes for every `--workers`. `per_throw.csv`: the decision's fields, `stages_*`, `catch_time_s`, `plans`,
`rejected`, `arrival_s`, `crossing_s`, `margin_s`, `lateral_m`, `angle_deg`, `clear_environment_m`, `clear_self_m`,
`clear_exempt_m`, `poses`, `certified`, `within_limits`, `at_rest`, `intercepted`, `tracking_m`, `tracking_joint_rad`,
`force_limit_ticks`, `control_limit_ticks`, `refused`; moving adds `plan_mode`, `stroke_scale`, `stroke_start_s`,
`stroke_end_s`, `cup_speed_m_s`, `relative_speed_m_s`, `direction_deg`, `in_stroke`. `summary.csv` (the rates,
`r2_violations`, the miss, margins, clearances, tracking; moving adds the strokes and speeds), `margins.csv`,
`manifest.json`. `interception_ledger.csv`: a closed record of 1 row.

### 9.14 The catch's files and its ledger (generated; F11)
R2 (one row and class per throw, acceleration and contact; no robot contact but the ball) and a moving catch's R1 are
checked; a violation is named, recorded, exit 1. Validation: the nominal contact for every seed, `Contact.Levels` for
the first; test: the stored margins, the nominal contact, from a committed tree after the validation `catch`; **every
test call is counted** by a row appended first to `catch_ledger.csv`. `derive.py sweep --run DIR --evaluations DIR`
(validation, not counted): the stored real-flight errors reach the injected prediction only. `per_throw.csv`: `dataset`,
`flight`, `seed`, `acceleration_deg_s2`, `time_constant_s`, `damping_ratio`, `catchable`, `decision`, `intercepted`,
`catch_time_s`, `class`, `outcome`, `retained`, `at_rest`, `rest_s`, `first_contact`, `first_contact_s`,
`cup_contact_s`, `entered`, `entry_lateral_m`, `end_speed_m_s`, `penetration_m`, `robot_contact`,
`tracking_joint_rad`, `force_limit_ticks` (the sweep first adds `error_source`, `error_scale`; moving ends with
`plan_mode`, `stroke_scale`, `contact_speed_m_s`, `markers`). `summary.csv`: `retained_of_go` and
`retained_of_catchable` with their intervals, the classes, `at_rest`, the rest times. `manifest.json`.

### 9.15 The view's frames (generated)
`.venv/bin/mjpython scripts/catch.py view --run DIR (--index I | --random [--pick-seed S] | --new SEED [--aim
catch|anywhere]) [--compare RUN ...] [--theme dark|light] [--camera side|overview|cup|thrower] [--ghost] [--markers]
...` opens the window; `--frames DIR [--size WxH] [--fps F] [--overlay]` writes `frame_<number>.png` and `frames.csv`
(`frame`, `file`, `time_s`, `phase`, `samples`, `method`, `prediction_n`, `inference_ms`, `plan`, `plans`, `outcome`).
A test throw is refused from the manifest before `flights.npz` is opened; the stored check compares a validation
throw's rows with the stored lines byte for byte (exit 1 on a difference). Nothing is counted.

### 9.16 The figures (generated)
`scripts/plots.py {training,comparison,prediction,decision,interception,catch,real,motion,all} [--generated DIR] [--out
DIR] [--theme dark|light] [--no-latex] [--flight NAME] [--pick-seed S] ...` reads only `Data/Generated/` by the files'
readers, the configuration and, for `real`, the 45 identification recordings; never a test per-flight file, a test throw
or a validation recording. **The full output**: `<out>/<name>_<theme>.pdf` and `.png`. **The key set**:
`<out>/README/readme_<role>_<theme>.png` for comparison_bars, prediction_error, prediction, real, catch and motion in
both themes, `<out>/Video/video_<role>_dark.png` (3840 × 2160) for comparison_bars and motion; the README's copies are
in `Media/`. **`captions.csv`** (`file`, `caption`): the source, the seeds, what is assumed; a real flight is said to be
an identification flight (a demonstration, not an independent evaluation) in the assumed frame; every caption ends with
the safety sentence. Exit 0 / 1 / 2; `all` 38–42 s with LaTeX.

## 10. Tests

`test/` mirrors `src/`. What is pinned: the port within 1e-9 m (all 12 000 throws slow); the recordings' counts, the 18
ambiguous steps, the 90 flights' lines, the 45 / 45 split; `identify` and `observation` end to end (slow); the
observation model's replay, bounds and estimator; the dataset byte for byte, the stable prefix, the partition sizes;
**leakage** (`train` and `evaluate` pass with the test-partition selector and every recording reader made to raise);
the trajectory contract (off-grid queries, the range refusal, query independence bit for bit, the end at the floor);
one seed's training bit for bit; the metrics and the comparison's bootstrap on constructed sets; the real-flight
predictions byte for byte (slow); the closed-form IK against a numerical one; the clearance, the lattice and the largest
cuboid against brute force; the `Areas.Catch` pin (slow); the scene without a warning, its fingerprint, the vendored
tree id; the ball within 0.1 mm / 1 mm/s and the 2 µm pin; the profile against Ruckig's durations; the stroke at 100
random cases; the catch chain's files from the validation partition only and the same for two workers; a counted test
read on generated data (one ledger row); a throw the cup at rest loses at 0.29 held by a stroke; removed sub-commands,
abbreviated options and removed predictors refused; a test throw refused before its file is opened; the theme changing
visual fields only; a stored throw equal to its stored rows (slow); every figure's plotted arrays against its fixture;
the contrast floor; the figure script under an audit of every file opened; the README's commands, numbers and alert.
**The viewer's window is not tested**: it is checked by eye. Expected numbers come from measurements, never from the
code under test. Per commit: `.venv/bin/python -m pytest -m "not slow" -n 8 --dist worksteal`; full: `.venv/bin/python
-m pytest -n 8 --dist worksteal`; before a release the per-commit tier also serially. Measured: per commit (pytest -m
"not slow" -n 8 --dist worksteal) 886 passed in 20.7 / 21.0 s in two runs; the full suite 928 passed in 350.2 s
(2026-11-24, Apple M2 Pro, on mains, macOS's background daemons quiet, the editor open). The earlier 16 s target of the
per-commit tier was dropped at v1.0.

## 11. Installer — `install.sh`

For macOS arm64 (verified): presence-check `/opt/homebrew/bin/python3.11` (never install it), create `.venv`, install
`requirements.txt`, then verify every file of `models/universal_robots_ur10e/` against `models/SHA256SUMS` with
`shasum`; a changed or missing file is named, the restoring command printed, exit 1. Nothing downloaded, nothing
repaired. Run for real on 2026-11-20: mujoco 3.16.0 installed, the 26 files matched.

## 12. (Removed)

Removed at v1.0: the development tooling is not part of the release.

## 13. Risks and how they are handled

| # | Risk | Handling |
|---|---|---|
| 1 | About half the in-flight frames are missing | The measured dropouts replayed; irregular input everywhere |
| 2 | Flight residual 5.4 / 6.6 mm against a 1.2 mm noise | No term without a held-out gain; the floor reported |
| 3 | The QTM-to-robot transform is unknown | Assumed, configurable (§7.3); every real-flight result says so |
| 7 | MuJoCo 3.16.0 lacks something needed | The work stops; no silent version change |
| 8 | Real flights outside the simulated throws | Coverage listed; 42 of 45 inside; results both ways |
| 9 | A learned predictor learns the 45 dropout patterns | Composed replay; no gain from knowing them measured (§4.8) |
| 10 | The test sets steer the selection | Every counted read in a ledger; selection on simulated validation only |
| 11 | MPS is not bitwise deterministic | The CPU is the reference device; an `mps` run refused by the test command |
| 13 | Real truth is noisy (0.61 mm) and gapped (up to 0.975 s) | Horizon ranges with counts; no catch metric |
| 14 | The query times leak the true flight's end | The trajectory is fixed before any query; tested |
| 17 | The learned predictor is level with GLS_M4, not better | Reported; real flights read by a rule fixed first |
| 18 | A query-time trajectory may never land | Refused as its own outcome; none on the real flights |
| 20 | The catch area is checked at lattice points | Shrunk one step; every motion certified along its path (§7.9) |
| 21 | The view is taken for the result | Stored throws compared byte for byte; display predictions never act |
| 22 | A stroke is a joint line, planned once | Certified like every motion; direction and speeds recorded |
| 23 | A figure is read as more than its source | Every number from a stored file; captions state the assumptions |

## 14. Open questions

1. **The camera mounts**: the 12 cameras at the rail's corners and at x = ±2.70…2.75 m, y = ±1.20…1.25 m, and the
   rail's height? Until answered: assumed, 3.0 m, centred above the flight area's centre (§4.4).
2. The ball's mass: a weighed value (until then 0.122 kg, assumed).
3. The robot base in the recordings' frame: a measured position (until then the transform of §7.3, assumed).
4. **The frame (decided)**: the assumed transform stays; `Areas.Throw` and `Areas.Flight` are at −y, as in 2022.
5. **Where F3's parameters live (decided)**: `Config/ball_catching.yaml` (`Flight.*`), pinned by a slow test.
6. **The floor height**: a trajectory, predicted or true, ends where the ball centre reaches z = 0.030 m. **The origin
   is on the floor under the robot's base axis**, the mounting face 0.180 m above it (pallet 144 mm, plate 16 mm,
   mounting plate 20 mm). Assumed until measured: that the recordings' z = 0 is the floor and the stand is the CAD's.

## 15. Sources

- The author's presentations (2021–2022, not in the repository); Qualisys "5+, 6+ and 7+" product information
  (2018-11-23) and the super-spherical marker sheet; the UR10e technical specification and UR's support article
  "Important deployment points" (read 2026-11-09).
- MuJoCo (github.com/google-deepmind/mujoco) and MuJoCo Menagerie `universal_robots_ur10e`, its LICENSE and README.
- The measurements of §4; the CAD in `CAD/`; the 2022 Blender areas; the view's reference diagram by the author.
