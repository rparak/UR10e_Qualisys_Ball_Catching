# Changelog

Every output of this project is a simulation result; nothing is run on, or validated on, the real UR10e.

## [1.0] - 2026-11-25

The first release. The numbers are in [docs/SPEC.md](docs/SPEC.md) §4, the open limitations in
[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

### Added

- **The 2022 code, ported** (F1): the flight model with its four integrators, the areas and the 2022 prediction
  pipeline as a test reference, reproducing the 2022 baseline within 1e-9 m per throw.
- **The recording loader** (F2): the 90 Qualisys recordings (100, 200 and 300 Hz) read with the time rebuilt from
  the frame count, every gap and irregular step reported, the main free flight segmented, the flights split 45 / 45
  into identification and held-out validation flights.
- **The flight model** (F3): gravity with quadratic drag, one coefficient k identified on the identification flights
  and validated on the held-out ones.
- **The observation model and the datasets** (F6): the motion-capture error (white noise, an in-run drift, an offset
  that moves while the ball is lost) and the recorded dropouts; simulated throws generated reproducibly from a
  configuration and a seed.
- **The predictors** (F7, F8, F14): `GLS_M4`, a physics fit weighted by the observation model, and `QT_BASIS`, a
  learned query-time model trained in PyTorch and run in NumPy; both evaluated on simulated test throws and on the 45
  held-out real flights, every counted read of held-out data in a ledger.
- **The scene and the workspace** (F4): the MuJoCo scene from the CAD with the vendored UR10e (MuJoCo Menagerie), a
  closed-form inverse kinematics, the catch area and the robot's ready pose.
- **The MuJoCo ball** (F5): the identified flight model applied in MuJoCo, agreeing with the reference integrator
  within 1 µm in free flight.
- **The catch decision** (F9): a conservative go / no-go from the prediction's margins.
- **The interception planner** (F10): the decision at every sample, time-optimal joint motions re-planned as samples
  arrive, every path certified collision-free by exact distances.
- **The physical catch** (F11): every go simulated with contact and classified, with a robustness sweep.
- **The moving catch** (F15): the robot meets the ball in motion on a stroke along its path; the static catch stays
  selectable.
- **The view and the figures**: `scripts/catch.py view` plays one throw in MuJoCo's viewer or as PNG frames;
  `scripts/plots.py` draws the figures of the stored results in a dark and a light theme.
- **The documentation** (F13): the README, `docs/` (the specification and the known issues),
  `CITATION.cff`.
