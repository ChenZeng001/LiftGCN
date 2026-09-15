# LiftGCN

Research code for **LiftGCN: Efficient Energy-Preserving Graph Learning via Joukowski Spectral Lifting for Finite Element Stress Prediction**.

**Paper link:**  [arXiv](https://arxiv.org/abs/2609.14977)

LiftGCN predicts nodal von Mises stress on finite element meshes. It uses second-order Joukowski spectral propagation with learnable per-channel coefficients shared across depth. Each propagation step combines one sparse graph multiplication with a node-wise nonlinear residual correction. The final regression layer uses the two most recent hidden states.

The energy-preserving interpretation applies to the linear Joukowski backbone; the complete model includes learned nonlinear corrections. Inputs contain three normalized spatial coordinates and one normalized log-degree feature. Loads are fixed within each dataset and are not included as input features.

This repository contains the model, data loading and graph construction utilities, training and evaluation code, and Abaqus simulation/export scripts for the Connecting Lug and Elbow Bracket datasets. Dataset files are distributed separately.

## Project structure

```text
code_github_share/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- models.py
|-- dataloader.py
|-- train.py
|-- datasets/                         # Dataset placeholder; no data files included
`-- simulation/
    |-- connecting_lug/
    |   |-- gsi_connecting_lug_parametric_analysis.py
    |   |-- run_connecting_lug_parametric.bat
    |   `-- connecting_lug_datasets_process.py
    `-- elbow_bracket/
        |-- gsi_elbow_bracket_parametric_analysis.py
        |-- run_elbow_bracket_parametric.bat
        `-- elbow_bracket_datasets_process.py
```

Git does not track empty directories. A `.gitkeep` placeholder preserves `datasets/` in the repository; it contains no dataset files. `.gitignore` excludes downloaded datasets, training outputs, and simulation result directories.

## Environment installation

Run the following commands from the repository root using a terminal with Conda available:

```bash
conda create -n liftgcn python=3.12 -y
conda activate liftgcn
python -m pip install -r requirements.txt
```

Training uses NumPy, pandas, PyTorch, and tqdm; PyTorch Geometric is not required. CPU execution is supported. For GPU training, install a CUDA-enabled PyTorch build appropriate for your system using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) before installing the remaining requirements. Verify the environment with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python train.py --help
```

Dependency ranges are provided rather than an exact environment lock. Full research runs can require substantial memory because graphs are cached and processed one at a time. Use `--no-cache` to reduce host memory usage or reduce `--hidden` for a smaller experimental model.

The simulation scripts run separately in **Abaqus/CAE 2025**, using its bundled Python runtime and an Abaqus/Standard license. The Conda environment is for machine learning and cannot supply the `abaqus`, `odbAccess`, or `caeModules` modules. Simulation scripts also import Pillow (`PIL`) to save contour images; it must be available in the Abaqus Python runtime. Downloaded CSV files can be used without Abaqus.

## Datasets

Download the datasets from [Google Drive](https://drive.google.com/drive/folders/1dMtMiSuOPAbmmXc3cwDgcxar2HPSx64E?usp=drive_link). Neither dataset is included in this repository.

| Dataset | Samples | Description |
| --- | ---: | --- |
| Connecting Lug | 225 | Parametric lug geometries under a fixed 20,000 N load |
| Elbow Bracket | 288 | Parametric curved bracket geometries under a fixed 2,000 N load |

Both datasets use C3D10M ten-node tetrahedral meshes and linear elastic material parameters (Young's modulus 200 GPa, Poisson's ratio 0.30). Each sample has its own mesh. Extract the downloaded folders to obtain:

```text
datasets/
|-- connecting_lug/
|   `-- connecting_lug_processed/
|       |-- input_coord/coord_1.csv
|       |-- input_matrix/matrix_1.csv
|       `-- output_stress/stress_1.csv
`-- elbow_bracket/
    `-- elbow_bracket_processed/
        |-- input_coord/coord_1.csv
        |-- input_matrix/matrix_1.csv
        `-- output_stress/stress_1.csv
```

Only the processed CSV directories are required for training. Raw CAE/ODB files, parameter records, and contour images may remain in the downloaded folders. Pass `--data-root` directly to the processed directory if your extraction layout differs.

| CSV category | Required columns | Meaning |
| --- | --- | --- |
| `input_coord/coord_i.csv` | `node,x,y,z` | Node labels and coordinates in meters |
| `input_matrix/matrix_i.csv` | `element,node1,...,node10` | Element connectivity in Abaqus local node order |
| `output_stress/stress_i.csv` | `node,von_mises` | Nodal equivalent stress in pascals |

Stress CSVs also contain `xx,yy,zz,xy,yz,zx`; this model reads only `von_mises`. The export scripts average element-nodal tensor contributions at each node, then compute von Mises stress from the averaged tensor. Connectivity tables are not adjacency or stiffness matrices. Numeric sample suffixes must match across the three directories, and stress rows are aligned to coordinates by node label.

## File usage

### `models.py`

Defines `JoukowskiBlock`, `StressJoukowskiGCN`, and the public alias `LiftGCN`. The model accepts an `[N, 4]` feature tensor and an `[N, N]` sparse symmetric normalized adjacency tensor. It returns `[N, 1]` predictions in standardized stress units.

```python
from models import LiftGCN

model = LiftGCN(in_dim=4, hidden=500, layers=6, dropout=0.1)
# pred_normalized = model(graph.x, graph.adj)
```

### `dataloader.py`

Discovers matching CSV triplets, validates node labels and element connectivity, computes normalization statistics using training samples only, and constructs cached `FEMGraphDataset` objects. Coordinates use training-set mean/std; log-degree is standardized within each graph. Target stress mean/std are calculated from training nodes.

C3D10M connectivity uses twelve half-edge segments per element, followed by cross-element deduplication and conversion to a bidirectional graph. Adjacency normalization does not add self-loops. Rows with exactly ten valid `node1` through `node10` entries are interpreted as C3D10M. Other element sizes use complete within-element connectivity; this fallback does not infer their physical edge topology.

```python
from pathlib import Path
from dataloader import discover_samples, collect_train_stats, FEMGraphDataset

root = Path("datasets/connecting_lug/connecting_lug_processed")
ids, files = discover_samples(root)
train_ids = ids[:int(0.8 * len(ids))]  # Illustrative split; train.py shuffles by seed.
stats = collect_train_stats(train_ids, files)
dataset = FEMGraphDataset(train_ids, files, stats)
graph = dataset[0]
```

### `train.py`

Train either dataset from the repository root:

```bash
python train.py --data-root datasets/connecting_lug/connecting_lug_processed --output-dir outputs/connecting_lug
python train.py --data-root datasets/elbow_bracket/elbow_bracket_processed --output-dir outputs/elbow_bracket
```

For a short execution check:

```bash
python train.py --data-root datasets/connecting_lug/connecting_lug_processed --epochs 1 --repeats 1 --hidden 32 --layers 2 --device cpu --output-dir outputs/smoke
```

This short command still reads the complete dataset and is not a reproduction of the paper's results. The default device is the first CUDA GPU when available, otherwise CPU. Select a device explicitly using `--device cuda:0` or `--device cpu`.

| Option | Default | Purpose |
| --- | --- | --- |
| `--data-root` | `datasets/elbow_bracket/elbow_bracket_processed` | Processed input directory |
| `--output-dir` | `outputs` | Checkpoints and run records |
| `--epochs` / `--repeats` | `50` / `10` | Training duration and independent runs |
| `--hidden` / `--layers` | `500` / `6` | Hidden width and propagation depth |
| `--dropout` | `0.1` | Dropout probability |
| `--lr` / `--weight-decay` | `0.002` / `0.00001` | AdamW settings |
| `--grad-clip` | `1.0` | Gradient norm clipping |
| `--train-ratio` / `--seed` | `0.8` / `42` | Sample-level split and initial seed |
| `--no-cache` | Disabled | Construct graphs on demand |

Each run uses a shuffled 80/20 sample-level split and seed `42 + repeat_index - 1`. Training minimizes MSE on standardized stress using AdamW and cosine learning-rate decay. Evaluation reports loss, NMSE, SNR, R2, spectral NMSE, hotspot PSNR, and hotspot stress-gradient NMSE at 1%, 5%, and 10%.

**Evaluation protocol:** To preserve the supplied experiment's behavior, the epoch with the lowest loss on the held-out 20% is selected and reported on that same split. The code calls this split the test set, but it also serves as the model-selection set. For an independent final test estimate, introduce a separate validation split before comparing methods. The original exhaustive CUDA latency benchmark is omitted from this training entry point.

Outputs are `config.json`, `metrics.json`, and `repeat_01.pt`, `repeat_02.pt`, etc. Each checkpoint includes model weights/configuration, normalization tensors, split IDs, seed, selected epoch, and metrics. Use a different output directory for each experiment; matching output filenames are overwritten on reruns.

To restore a checkpoint and predict physical stress for another graph with compatible CSV files:

```python
import torch
from models import LiftGCN
from dataloader import FEMGraphDataset

checkpoint = torch.load("outputs/connecting_lug/repeat_01.pt", map_location="cpu", weights_only=True)
model = LiftGCN(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
stats = {key: value.numpy() for key, value in checkpoint["stats"].items()}
# ids and files come from discover_samples(root), as shown above.
graph = FEMGraphDataset(ids, files, stats)[0]
with torch.no_grad():
    pred_pa = model(graph.x, graph.adj) * checkpoint["stats"]["stress_std"] + checkpoint["stats"]["stress_mean"]
```

### `simulation/`

Each dataset directory contains three standalone files:

| File | Usage |
| --- | --- |
| `gsi_*_parametric_analysis.py` | Build geometry, assign material and boundary conditions, mesh, solve, and save CAE/ODB files and contour images using Abaqus/CAE |
| `run_*_parametric.bat` | Run the parameter grid from Windows, collect sample outputs, and write `parameters.txt` |
| `*_datasets_process.py` | Export coordinates, connectivity, forces, displacements, and stresses from CAE/ODB samples to CSV |

In a **Windows Command Prompt with `abaqus` on PATH**, run the full parameter grid:

```bat
cd simulation\connecting_lug
run_connecting_lug_parametric.bat
abaqus cae noGUI=connecting_lug_datasets_process.py -- --input-dir connecting_lug --output-dir ..\..\datasets\connecting_lug\connecting_lug_processed
cd ..\..

cd simulation\elbow_bracket
run_elbow_bracket_parametric.bat
abaqus cae noGUI=elbow_bracket_datasets_process.py -- --input-dir elbow_bracket --output-dir ..\..\datasets\elbow_bracket\elbow_bracket_processed
cd ..\..
```

The launchers generate 225 Connecting Lug cases and 288 Elbow Bracket cases with the provided parameter grids. Raw samples are saved under `simulation/connecting_lug/connecting_lug/` or `simulation/elbow_bracket/elbow_bracket/`. They use a temporary `results/` directory and clean it after collecting results. Run them in their supplied directories and keep unrelated files out of `results/`.

For a limited first run, set `LUG_MAX_RUNS=1` or `ELBOW_MAX_RUNS=1` in the same Command Prompt before calling the relevant launcher. Clear the variable afterward with `set LUG_MAX_RUNS=` or `set ELBOW_MAX_RUNS=`. These limits start from the beginning of the parameter grid; rerunning is not a parameter-aware resume operation.

The simulation Python files also support individual analyses via `abaqus cae noGUI=...`; see their top-level documentation and parameter definitions. Export options include `--instance`, `--model`, `--step`, `--frame`, and `--fail-fast`, passed after `--`. The exporters additionally write `input_force/` and `output_displace/`; these outputs are not required by LiftGCN and are not included in the released processed datasets.

## Citation

If you use this code or the datasets, please cite the accompanying paper. The BibTeX entry will be added when publication details are available.

```bibtex
@misc{zeng-2026-liftgcn,
	author = {Zeng, Chen and Wang, Qiao},
	month = {9},
	title = {{LiftGCN: Efficient Energy-Preserving Graph Learning via Joukowski Spectral Lifting for Finite Element Stress Prediction}},
	year = {2026},
	url = {https://arxiv.org/abs/2609.14977},
}
```
