# Jet_Ana_AI

A runnable research pipeline for **contextual embeddings of EMCal clusters**, followed by variable-count grouping into EMCal-visible jet candidates and unassigned activity. It uses PyTorch message passing, without requiring PyTorch Geometric. Real-data training does not require simulated ancestry labels.

**Status:** implementation and software tests, not an experimentally validated jet finder. Unlabeled training learns representations; it does not establish that embedding neighbors share a parent. No pretrained physics model is supplied. All default angular/energy scales and perturbations are starting settings to tune on detector-specific controls.

## Quick start

Run commands from this folder. A local `.venv` has already been created in this workspace. For a fresh installation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[root,test]'
```

You can also use `python -m jet_ana_ai.cli` without installing the editable package, provided the dependencies are installed and you are in this folder.

### Run on real data

Create a CSV with one row per cluster:

```csv
event_id,cluster_id,energy,eta,phi
run42_evt123,0,4.2,0.12,1.31
run42_evt123,1,1.3,0.16,1.36
```

- `event_id`: globally unique string, preferably including run ID.
- `cluster_id`: unique within the event.
- `energy`: positive cluster energy in GeV.
- `eta`, `phi`: direction relative to the reconstructed event vertex, phi in radians. Phi is wrapped internally. Use properly calibrated, quality-selected clusters.
- Optional numerical measurements: e.g. `time`, `n_cells`, `m02`, `m20`, `max_cell_fraction`. Add their names to `optional_features` in the configuration. Missing/nonfinite inputs are rejected, not silently imputed. Convert detector timing/shape conventions explicitly.
- Optional `label`: `-2` unknown (default), `-1` trusted background, nonnegative integer for a trusted event-local jet group. Group IDs have no cross-event identity. Do not add labels, IDs, reference jet variables, or predictions as features.

Create an event registry CSV to retain empty events and control splits:

```csv
event_id,split,split_group
run42_evt123,train,run42
run43_evt456,val,run43
run44_evt789,test,run44
run44_evt790,test,run44
```

An event with no cluster rows is preserved by this registry. If omitted, only observed events are known. Without explicit `split`, a seeded hash of `split_group` (or event ID) assigns 70/15/15 percent train/validation/test. Keep correlated events/runs together. A split group crossing partitions is rejected. For small data, explicitly provide enough nonempty train and validation events.

```bash
python -m jet_ana_ai.cli validate --data data/clusters.csv --events data/events.csv
python -m jet_ana_ai.cli train --data data/clusters.csv --events data/events.csv --mode self --epochs 30 --output runs/self
python -m jet_ana_ai.cli predict --data data/clusters.csv --events data/events.csv --checkpoint runs/self/best.pt --output runs/predictions
python -m jet_ana_ai.cli evaluate --predictions runs/predictions
python -m jet_ana_ai.cli plot --predictions runs/predictions --event-id run44_evt789 --output runs/predictions/event.png
```

`predict` selects the test partition by default. Set `--split val` for tuning; use test only for final assessment. `--split all` is useful for debugging, and outputs flag events present in checkpoint training/validation lists. This check detects repeated IDs, not re-exported events with changed IDs. Use consistent IDs and partition registries across files.

### Convert ROOT

Copy `examples/root_mapping.json` and replace the branch names with yours:

```bash
python -m jet_ana_ai.cli convert-root --input data/input.root --tree Events --mapping examples/root_mapping.json --layout jagged --output data/clusters.csv
```

This writes `data/clusters.csv` and `data/clusters_events.csv`. Use the second as `--events`; edit it to add splits/groups.

- `jagged`: one tree entry per event; scalar event ID; aligned arrays of energies, eta, phi, and optional cluster features. Cluster IDs are generated from array position unless mapped. Empty events are retained.
- `flat`: one entry per cluster; scalar branches; explicitly map `cluster_id`. Empty events are absent and need a separate registry.
- Mapping keys are standard output names and values are ROOT branch names. Optional labels/features can be included. Scalar event metadata other than event ID should be provided through a separate registry, not as jagged cluster features.
- IDs must be globally unique. If your file has separate run/event branches, construct their combined ID upstream. No expression evaluation or automatic unit conversion is performed.
- Reads are chunked but the converted output is assembled in memory; shard large files and process them with a deliberate, consistent split registry.

### Compare three methods

```bash
python -m jet_ana_ai.cli predict --data data/clusters.csv --events data/events.csv --method spatial --output runs/spatial
python -m jet_ana_ai.cli predict --data data/clusters.csv --events data/events.csv --method features --output runs/features
```

`spatial`: angular distance only. `features`: train-standardized engineered-feature distance plus geometry. `learned`: contextual embedding distance plus geometry. The feature baseline needs nonempty training data for its scaler; learned inference uses the scaler and feature schema saved in its checkpoint. All grouping settings come from the supplied config so you can vary them without retraining.

**Compare at matched background-control false-candidate rates**, not merely at identical numerical thresholds: feature/embedding distances have different scales.

## Model and training

Input features are log transverse energy, eta, neighboring ET and multiplicity at several radii, central energy fraction, capped nearest-neighbor distance, and optional detector measurements. Self-neighbors are excluded. ET is computed as `energy / cosh(eta)` under the massless cluster approximation.

The GNN builds angular nearest-neighbor edges within a configurable radius. Edges carry delta eta, wrapped delta phi, and delta R. Local message passing and event mean context produce one embedding per cluster. Cluster ordering has no intended significance. No absolute phi feature is used; detector-specific phi response still needs calibration and control studies.

### Unlabeled mode (`--mode self`)

Trains with masked feature reconstruction and a VICReg-style embedding objective: agreement between two mildly perturbed views, a variance floor, and covariance suppression. Matching views retain the same clusters; neighborhood summaries are recomputed after perturbing energy/direction. Masks hide node features but not graph geometry. The loss is an exploratory adaptation, not a demonstrated optimal objective for jet membership.

The signal/background head is **not used** in self mode; `signal_score` is blank. Self-supervised embeddings may retain nuisance information or organize by energy rather than jet membership. Validation loss selects representation quality according to this objective, not physics performance. Variance regularization is event-local and can spread background embeddings too; inspect and ablate it.

Default jitter is not a detector-resolution model. Tune it to measured uncertainties or set the jitter values to zero. No global rotations/crops or artificial shower splittings are imposed on a detector with restricted acceptance.

### Optional weak labels (`--mode weak`)

Adds binary jet-associated/background loss for known labels and a contrastive distance loss for labeled pairs. Same-group pairs attract; different groups or jet/background pairs repel. Unknown clusters are excluded from these losses; background/background pairs are never treated as a shared object. Requires both trusted signal and background labels in training. Unlabeled losses still use all clusters.

Labels can come from independent data controls; do not fabricate per-cluster truth from event-level enrichment alone. Conventional-jet labels induce conventional-jet biases. The optional classifier output is uncalibrated relative to those weak labels, not an ancestry probability.

## Grouping and outputs

Inference uses DBSCAN on a hybrid distance:

```
d² = (delta_R / spatial_scale)²
     + embedding_weight² * (RMS_embedding_difference / embedding_scale)²
```

Pairs outside `max_link_dr` cannot link. Optional weak-label scores filter nodes before grouping. Minimum cluster count and scalar ET sum reject small groups. There is no fixed jet count; zero candidates is allowed. Candidates are ordered by descending scalar ET sum within each event.

Density connectivity can bridge nearby objects: `max_link_dr` limits individual links, not group diameter. Inspect `max_centroid_dr`, splitting/merging, and dense-background controls. Peripheral clusters can stay unassigned. This version returns hard group IDs plus a heuristic affinity, not normalized soft memberships over competing jets.

Outputs:

- `clusters.csv`: original measurements, `candidate_id` (`-1` unassigned), `membership_affinity` (distance-based, uncalibrated), optional `signal_score`.
- `candidates.csv`: counts, scalar cluster ET sum, energy-weighted eta/circular phi centroid, angular RMS and maximum centroid distance. **These are not calibrated full-jet four-vectors or pT.**
- `events.csv`: all selected events including empty ones; candidate counts, unassigned counts, partition and checkpoint-overlap flags.
- `embeddings.csv`: event and cluster keys plus embedding coordinates, for learned/feature modes when nonempty.
- `metadata.json`: configurations and interpretation; `report.json`: diagnostics and optional reference-label metrics.

Unassigned does not prove background. A true jet represented by one cluster is not accepted by default. Changing that requires an independent isolated-photon/electron rejection strategy.

## Validation before physics use

1. Define target acceptance, ET scale, and what counts as background. EMCal alone does not measure all jet constituents.
2. Tune only on training/validation controls; freeze thresholds before examining the final test set.
3. Compare methods at matched false-candidate rates in independent background controls. Mixed events omit genuine correlations and are not automatically a complete background model.
4. Check recoil/track associations using independent information where possible; quantities already enforced in inputs/selection are not independent validation.
5. Study occupancy, detector edges/dead regions, time/run dependence, isolated EM objects, split showers, jet merging, peripheral membership stability, and changes under measurement uncertainties.
6. Evaluate embedding dimension/scale, graph radius, feature choices, training objectives and DBSCAN thresholds. Low self-supervised loss alone is insufficient evidence of useful jets.

Boundary-area corrections and event-background subtraction are **not** silently inferred here. Neighborhood features near detector boundaries need acceptance-aware treatment or fiducial cuts. Optional supplied background-density/quality measurements can be features. No production cross sections, calibrated probabilities, complete ancestry, or simulation-to-data corrections are claimed.

Memory: geometry/grouping are quadratic in event cluster count and embedded distances add a dimension factor. The default cap is 1500 clusters; oversized events fail rather than truncate. The loader retains input tables in memory. Large heavy-ion datasets may require sparse neighbor search, streaming/sharded datasets, and a detector-specific background model before scaling. CPU and CUDA are supported; `--device mps` is optional and unvalidated here.

## Software smoke test

Artificial patterns are provided **only to exercise the software**, not as training evidence for real physics:

```bash
python -m jet_ana_ai.cli demo --output examples/demo_data --events 60
python -m jet_ana_ai.cli train --data examples/demo_data/clusters.csv --events examples/demo_data/events.csv --mode self --epochs 3 --output runs/demo
python -m jet_ana_ai.cli predict --data examples/demo_data/clusters.csv --events examples/demo_data/events.csv --checkpoint runs/demo/best.pt --output runs/demo_predictions
python -m jet_ana_ai.cli evaluate --predictions runs/demo_predictions
python -m pytest -q
```

The artificial labels are ignored by self-mode training; evaluation reports them as artificial reference labels. Run a weak-mode smoke test by changing `--mode self` to `--mode weak` and using a different output directory.

## Sources

- GravNet: https://arxiv.org/abs/1902.07987
- Object condensation (related concept, not implemented loss): https://arxiv.org/abs/2002.03605
- VICReg: https://arxiv.org/abs/2105.04906
- PyTorch AdamW: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW
- DBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN

See `docs/DESIGN.md` for an implementation map. `requirements-tested.txt` records the tested environment; the broader package requirements support fresh installations on other platforms.
