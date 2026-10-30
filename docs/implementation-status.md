**Private snapshot scope — 2026-10-30**

This snapshot preserves source for the HumanML3D backbone, contrastive VQ, GPT, cached-feature length estimator, single-segment length-constrained sampling, and latent-space analysis. Packaging checks do not constitute a fresh training run or verification of historical experiment outputs.

| Component | Source included | Limits |
|---|---|---|
| VQ/GPT backbone | Yes | Data, evaluation assets, weights, normalization and token caches must be legally obtained and configured. |
| TMR-guided VQ | Yes | Requires caption-aligned TMR cache and model assets; source/export provenance and differences from the target paper need further review. |
| Length estimator | Yes | Existing working code and training entry included. No claim that the architecture/features exactly match every detail of the paper. |
| Single-segment duration control | Yes | Default limit is 49 motion tokens / 196 frames. Explicit clipping is not arbitrary-length generation. |
| Transition model | Model class only | A working transition training/inference pipeline is not provided. The misleading legacy `train_transition.py` is excluded. |
| Complete arbitrary-length AMG pipeline | No | Segmentation, transition integration and full evaluation remain incomplete. |
| BABEL-TEACH / KIT paper tables | Not validated by this preparation | Historical branches and resources are not evidence of reproduced results. |
| Latent-space figures | Source and configuration only | Outputs, checkpoints and embedded motion review data are excluded; selections and figure-specific dependencies must be reconstructed. |

Old experiment launchers encode different VQ/GPT batches, steps and restore rules. Directory names do not establish the actual batch. Read script arguments and manifests before use. Formal results must identify the exact source commit, config, split, VQ/GPT pairing, evaluator, seeds and statistics. No paper metric is reported as a newly measured result in this snapshot.

Public release still requires contributor/IP authorization, complete third-party provenance, clean-environment installation, licensed resource acquisition, a runnable minimum demo, and verification of every reproduction claim. This private repository must not be submitted as an anonymous artifact without a separate anonymization process.
