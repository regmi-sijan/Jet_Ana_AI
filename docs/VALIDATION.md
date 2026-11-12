# Software validation

Validated locally on Apple Silicon, Python 3.14, CPU. Exact installed dependency versions are recorded in `requirements-tested.txt`.

- 12 automated tests passed: self and weak end-to-end training/checkpoint/inference/evaluation; spatial and engineered-feature baselines; periodic phi geometry; permutation equivariance; global phi-shift consistency; singleton inputs; empty events; variable group counts and rejection; duplicate/nonfinite input rejection; event-group split protection; optional-feature label-leak rejection; flat and jagged ROOT conversion; finite weak-loss gradients for identical embeddings.
- A separate 40-event artificial-pattern CLI run completed three self-supervised epochs. Validation objectives were finite throughout; these values measure only the demonstration objective.
- Learned, spatial, and engineered-feature inference commands completed on the held-out partition.
- Plot rendering and output report generation were exercised.

Artificial patterns do not model a detector, hadronization, underlying-event correlations, or measured jet performance. No real EMCal data were supplied. GPU/MPS execution, detector acceptance corrections, physics calibration and independent real-data validation remain untested.
