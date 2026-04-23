# CMIR: Cross-Modal Information Retrieval

CMIR is a project focused on the analysis, IR (Information Retrieval) generation, and hardware scheduling for Cross-Modal models, specifically targeting the **JEDI-linear** architecture. This repository provides tools to convert Keras models into High-Level IR (NN-IR), decompose them into scheduled primitives (Sched-IR), and estimate hardware costs using `da4ml`.

## Project Structure

- `IR/`: Core logic for NN-IR and Sched-IR generation.
  - `NN-IR/`: High-level Graph representation from Keras.
  - `Sched-IR/`: Primitive decomposition and scheduling engine.
  - `main.py`: Main entry point for the analysis and web viewer.
- `JEDI-linear/`: Sub-project containing the model definitions and training logic.
- `heterograph/`: Graph visualization library used for the IR web viewer.
- `official_models/`: Directory for storing pretrained model weights.

## Setup Instructions

### 1. Clone the Repository
Ensure you have the full project structure including the `JEDI-linear` and `heterograph` submodules/directories.

```bash
git clone https://github.com/SriV04/CMIR.git
cd CMIR
```

### 2. Create Conda Environment
The project requires a specific environment with dependencies like `da4ml`, `HGQ2`, and `graph-tool`.

```bash
# Create the environment from the JEDI-linear environment file
conda env create -f JEDI-linear/environment.yml -n jedi-linear

# Activate the environment
conda activate jedi-linear
```

Key dependencies included:
- `python=3.13`
- `da4ml < 0.5`
- `HGQ2` (for quantized layers)
- `jax` & `keras`
- `graph-tool` (for IR graph processing)

### 3. Setting up da4ml
`da4ml` (Distributed Arithmetic for Machine Learning) is used for hardware cost estimation. Ensure it is correctly installed within your conda environment (this is handled by the `environment.yml` above).

## Running the IR Analysis

The main analysis pipeline builds the NN-IR, performs scheduling, and launches a web-based viewer.

```bash
# Set the Keras backend to jax and run the main IR script
KERAS_BACKEND=jax python IR/main.py
```

Once running, you can view the interactive IR graphs and Gantt charts at:
**[http://localhost:8888](http://localhost:8888)**

The viewer provides several side-by-side comparisons:
- **NN-IR**: The raw model graph.
- **Sched BIND**: The unscheduled primitives.
- **Sched K=1/K=4**: Scheduled outputs for different fold factors.
- **Gantt Charts**: Cycle-accurate timing visualizations.

## Loading Pretrained Models

Pretrained weights are typically stored in the `official_models/` directory (extracted from `official_models.tar.gz`).

To load weights into a model:
1. Initialize the model configuration (e.g., `n_constituents=8`).
2. Build the model architecture using `get_gnn(conf)`.
3. Use `model.load_weights(path_to_keras_file)`.

Example:
```python
from JEDI-linear.src.model import get_gnn
from types import SimpleNamespace

conf = SimpleNamespace(n_constituents=8, pt_eta_phi=True)
model = get_gnn(conf)
model.load_weights("official_models/jedi_linear_8p.keras")
```

## Hardware Resource Analysis
The scheduling engine uses `IR/Sched-IR/da4ml-resource.yaml` to define the target hardware (e.g., Xilinx VU13P) and the cost functions for the `da4ml` estimator. You can modify this file to target different devices or resource constraints.
