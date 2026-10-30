**Resources excluded from the code snapshot**

| Resource | Expected location | Source / handling |
|---|---|---|
| HumanML3D motion features, captions and official splits | `dataset/HumanML3D/` | Obtain through https://github.com/EricGuo5513/HumanML3D following AMASS and constituent data licenses. Do not rehost the local dataset. |
| Evaluator, GloVe and normalization | `checkpoints/`, `glove/` | See https://github.com/Mael-zys/T2M-GPT and the original text-to-motion evaluator project https://github.com/EricGuo5513/text-to-motion . Verify the selected versions, license, paths and hashes. |
| CLIP ViT-B/32 | `ViT-B-32.pt` | Obtain through the pinned CLIP package / official model source; do not embed a personal signed URL. |
| TMR model and statistics | `models/Retrival/models/tmr_humanml3d_guoh3dfeats/last_weights/`, `models/Retrival/stats/` | Follow https://github.com/Mathux/TMR . Verify weight/statistics identity separately; a matching filename is insufficient. |
| TMR annotations, token indices and embeddings | `models/Retrival/datasets/annotations/humanml3d/` | Required by the inherited feature preparation workflow; not included. Official preparation or a verified replacement procedure remains to be documented. |
| DistilBERT | `output/amg_prepare/models/distilbert-base-uncased/` or explicitly configured path | Existing preparation entry uses a local model; weights/tokenizer and their version/license need separate acquisition. |
| Derived lengths and CLIP graph features | `output/amg_prepare/` | Regenerate from legally obtained data; review audit exclusions and the exact split protocol. |
| Trained VQ/GPT/length models and identity manifests | `output/hml3d/` | Not distributed. Freeze matching model identities and record SHA256 before any inference or evaluation. |

The absence of these assets is intentional. No external model URLs or SHA256 values are fabricated. Download provenance and hashes for the actual assets should be added when verified; private local experiment weights are not published by this preparation step.

The inherited dataset download scripts are excluded from this initial candidate because they contain hard-coded proxy settings, destructive directory replacement and inconsistent extraction paths. Follow official instructions in an empty resource workspace; do not run those legacy scripts against an active experiment tree.

HumanML3D input is `(T,263)` motion features at 20 fps for the selected workflow; generated `(T,22,3)` skeleton coordinates are not a substitute for training features. Train, validation and test splits must remain separate. Normalization, VQ, token caches, GPT and evaluator must be mutually compatible.

For length labels, `scripts/prepare_amg_lengths.py` writes `output/amg_prepare/lengths` by default. If the selected protocol uses `lengths_full_v2`, pass the corresponding `--output` explicitly and inspect the audit instead of assuming historical caches are present. The full training launcher similarly assumes prepared resources; it is not a bootstrap installer.
