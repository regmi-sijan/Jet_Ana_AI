# Implementation map

| Module | Responsibility |
|---|---|
| `data.py` | Schema validation, event registries, leakage-resistant splits, circular angles, multi-scale features, train-only scaling |
| `model.py` | Local graph message passing, event context, embeddings/reconstruction/optional node head, self/weak objectives |
| `train.py` | Recomputed augmented views, node masking, AdamW, validation checkpoint selection, checkpoint loading |
| `grouping.py` | Hybrid angular/embedding distance, optional signal filtering, DBSCAN, candidate cuts and summaries |
| `inference.py` | Learned/spatial/feature baselines, output tables and run metadata |
| `root_io.py` | Explicit flat/jagged ROOT-to-CSV conversion |
| `evaluate.py` | Candidate/background diagnostics, optional known-label metrics, eta-phi plots |
| `demo.py` | Artificial software-test patterns with 0–3 groups and empty events |
| `cli.py` | Command-line entry points and configuration validation |

One event is one optimizer step. Events have equal training weight, irrespective of cluster count. There is no padding or silent cluster truncation. Pairwise weak supervision is balanced by averaging positive and negative terms within an event; the node binary loss is not automatically class-balanced. Inspect imbalance before using weak labels in physics data.

A basic configuration is JSON; model/schema/scaler and training/validation IDs are saved in `best.pt`. During learned inference the checkpoint determines the feature schema and architecture, while the supplied configuration determines grouping. Checkpoints contain tensors and primitive types and are loaded with `weights_only=True`; load trusted artifacts.

The model does not predict the number of jets directly. The grouping stage selects density-connected candidates, rejects unsupported groups, and retains unassigned nodes. It does not infer a parton ancestry tree. Hard membership is accompanied by an affinity to another core/member in the group; that value is not a calibration or an estimate of peripheral truth purity.

Useful next extensions after establishing real-data controls: acceptance-aware local features, background-conditioned embeddings, sparse scalable graph search, independently supervised pair affinity, calibrated soft group assignments, and object-condensation training when credible instance labels become available.
