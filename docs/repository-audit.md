> Subsequent `tools/` review removed obsolete launch presets and one-off diagnostics, consolidated the GPT launcher, and moved required validators into `tools/checks/`. See the [tool index](../tools/README.md). File counts and removal totals in the initial privacy review below identify that review revision.

# Repository privacy and content audit

Date: 2026-10-31. Base revision: `adc53c924844a9c65d52e82f759d5f883a9c2ff5`.

This review covers the current `main` file tree: source, documentation, configuration, dependency history, and the three README figures. The initial tree contained 413 files. The accompanying [audit record](repository-audit.json) lists exact cleanup decisions without reproducing private path strings.

## Privacy findings and changes

- Legacy absolute paths containing a personal machine account were present in the local source and even in the path-migration helper. They are replaced with generic project/data/spaCy placeholders. The helper now maps those placeholders to the user's chosen checkout and data paths.
- No access-token, API-key, private-key, or hard-coded password signature was found by the text-pattern checks. This is a bounded source scan, not a guarantee against every possible secret format.
- The figures contained EXIF/text metadata chunks. Those chunks were removed from all three images while preserving the compressed pixel data and color information. Visual inspection found synthetic motion figures and captions, with no personal account details in the visible images.
- Remaining email-like strings are documented server examples and retained third-party attribution/license contacts. Copyright and attribution text remains intact. Project repository links still identify the GitHub owner and project.

## Removed material

Twenty-two files were removed:

| Category | Count | Examples / reason |
|---|---:|---|
| Identical or empty entry points | 2 | `train_transition.py` duplicated the GPT trainer; `train_GPT_split.py` was empty |
| Ad hoc chart/sample/result scripts | 3 | Hard-coded metric chart, selected-sample copier, fixed-result-array inspector |
| Obsolete command notes | 1 | `run_order` contained absent commands and inconsistent arguments |
| Obsolete resource download scripts | 5 | Destructive resource replacement, fixed proxy, or inconsistent extraction paths; official acquisition links remain |
| Standalone test files | 6 | Tests outside the requested source-delivery scope; no production entry point imported or invoked them |
| Stale check/environment outputs | 4 | Previous packaging/check reports and a redundant package inventory, replaced by this audit and the dependency history |
| Inactive duplicate configuration | 1 | `.gitignore.original` was not applied by Git |

The original source identities remain in the source manifest. Removed entries have `status: removed_by_cleanup`, a reason, and no active destination. The canonical GPT trainer remains `reproduction/local/train_t2m_trans.py`. Empty package `__init__.py` files are retained where they define package structure.

## Code outside the training loop

| Retained group | Purpose |
|---|---|
| `preprocess/`, TMR encoders, resource/configuration helpers | Build training inputs and configure legally acquired external resources |
| Evaluation tools and test-split launchers | Calculate project evaluation metrics; these are distinct from standalone unit/probe tests |
| Generation, reconstruction, preview, GIF/video conversion | Run trained models and inspect outputs |
| Latent-space analysis and rendering source | Reproduce representation analyses and figures; these scripts do not embed motion arrays or generated result files |
| `reproduction/local/LibMTL/` | 41 Python files of historical optional research source; no static import from the project's training entry points was found |
| Original length trainer | Historical model-related source with a pre-existing missing `utils/utils.py` dependency; use the maintained root-level length workflow for the documented executable pipeline |

The main and local-reproduction layouts share **62 groups of byte-identical files**. These are predominantly models, utilities, and TMR configuration/source. They are retained because both layouts have independently documented imports and working-directory conventions. Removing one copy would require a source-layout refactor and validation of both workflows. Git stores identical blob content once. The original local source remains the baseline for archived local experiments.

## Results and assets

No motion arrays, datasets, captions, split files, embeddings, normalization tensors, model weights, archives, generated videos/GIFs, or user-study outputs are tracked in the current tree. The three PNG figures are intentional README assets; their pixels are preserved and hidden EXIF/text metadata has been removed. JSON protocol/configuration files remain as analysis inputs, rather than generated motion or metric output files. The root ignore rules now also exclude common data/result extensions.

## Verification

The audit checks current personal-path and credential patterns, all retained Python/JSON/shell syntax, source-manifest identities, PNG chunk integrity, and absence of prohibited data/result binaries. The path configuration helper is exercised in a temporary copy: dry-run must not write, configured paths may contain spaces/quotes, written Python must parse, and a repeated write must be idempotent. Full model training, a fresh dependency installation, and paper-metric evaluation are outside this check.

## Git history and other branches

This cleanup changes current `main` files only. Earlier commits still contain earlier path strings and image metadata, and Git commit author/committer identities remain part of repository history. The separate `dev` branch is an older revision and is not updated by this cleanup. A cleaned current file tree is not an anonymized repository or a history purge. No history rewrite, force push, or branch deletion is performed here.
