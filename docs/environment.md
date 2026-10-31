# Environment and installation

AMG-T2M has one active Python dependency specification: the repository-root [`requirements.txt`](../requirements.txt). It contains the training stack, TMR/text encoders, graph-based length prediction, the English spaCy model, evaluation, latent-space analysis, Python rendering dependencies, and the retained TMR web interface. Direct dependencies are grouped by purpose; resolved transitive versions are pinned in the final section of the same file.

## Supported training target

The unified dependency set is resolved for **Python 3.10, Linux x86_64, and PyTorch 2.0.1 with CUDA 11.7**. The recorded interpreter is Python 3.10.18. GPU training requires an NVIDIA driver compatible with the selected CUDA runtime; the wheels provide the Python-side runtime. See the [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) for that release.

```bash
conda env create -f environment.yml
conda activate AMG-T2M
python -m pip install -r requirements.txt
python -m pip check
```

`environment.yml` only creates Python and pip. Package versions are maintained exclusively in `requirements.txt`; do not install a second TMR, length, visualization, or local-demo requirements file. Run this installation from the repository root before entering `reproduction/local/` for historical local-source commands.

A Python virtual environment can be used instead of Conda when Python 3.10 is already installed:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade 'pip==24.3.1'
python -m pip install -r requirements.txt
```

The CLIP source is pinned to a Git commit. The spaCy English model is pinned to a wheel URL and SHA256. Installation therefore needs Git and access to PyPI, the official PyTorch wheel index, and GitHub. The spaCy wheel is an NLP model dependency; motion datasets, body models, and experimental checkpoints remain external resources.

## What changed during consolidation

The original main environment and task overlays were separate installed-package inventories. They contained conflicting versions of shared packages, while the local demo additionally pinned PyTorch 2.5.1. The unified environment keeps the project's PyTorch 2.0.1/CUDA 11.7 model stack and selects a single compatible set for the other tasks. Full dependency metadata was resolved together for the supported Linux/Python target; it is not a concatenation of the old lock files.

The old inventories, minimal-demo requirements, nested TMR/visualization requirements, and historical local Conda export are preserved verbatim in [`dependency-history.json`](dependency-history.json). [`environment-versions.json`](environment-versions.json) remains a historical record of the original observed environment, rather than the current installation specification. The source manifest identifies archived dependency inputs and retains their original local hashes.

The length launcher now imports dependencies from the active environment and defaults to the installed `en_core_web_sm` package. It no longer requires `.runtime/length_deps` or `.runtime/length_models`. `T2M_SPACY_MODEL` still accepts another local spaCy model path, and an explicitly set `T2M_LENGTH_DEPS` remains available for an existing legacy overlay. Omit these overrides for the standard installation. `T2M_PYTHON=/path/to/python` can select a different interpreter for launchers.

The unused `import parser` was removed from the archived local `train_vq.py`, because that standard-library module is unavailable in Python 3.10. No training computation or hyperparameter was changed.

## Check the environment

```bash
python - <<'PY'
import torch, clip, transformers, spacy, torch_geometric
print('PyTorch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('GPU available:', torch.cuda.is_available())
print('Transformers:', transformers.__version__)
print('PyG:', torch_geometric.__version__)
print('spaCy pipeline:', spacy.load('en_core_web_sm').pipe_names)
PY
```

Most training launchers and length-controlled generation use CUDA explicitly. Skeleton preview can run on CPU. The dependency-resolution check covers the Linux training target; a complete installation or GPU experiment was not executed as part of this documentation change.

## Native rendering and optional extensions

The requirements file includes `smplx`, `pyrender`, `trimesh`, OpenGL Python bindings, video IO, and related packages. Mesh rendering also needs separately acquired SMPL/SMPLify model and prior files plus a system OpenGL backend. The retained `render_final.py` selects OSMesa, so its host needs a working OSMesa installation; skeleton GIF preview uses Matplotlib/Pillow and does not require SMPL or OSMesa.

The optional `mesh_intersection` collision extension is imported only when that SMPLify feature is enabled. It is a separate compiled CUDA extension and must match the host's PyTorch/CUDA toolchain. The basic PyG graph layers do not require installing every optional PyG CUDA extension; see the [PyG 2.5.3 installation documentation](https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html).

Linux launchers also use Bash, `flock`, GNU `cp --parents`, and `sha256sum`. These are operating-system tools, not pip dependencies.
