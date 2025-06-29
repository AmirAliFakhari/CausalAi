# Causal Graph Discovery and Inference

This project provides a comprehensive framework for discovering and inferring causal relationships from observational data using various causal discovery algorithms. The pipeline is designed to be modular, reproducible, and compliant with domain-specific constraints.

## Project Overview

The `causation.ipynb` Jupyter Notebook implements a complete workflow for causal discovery, covering data preprocessing, feature engineering, model training, evaluation, and visualization. It applies a tiered constraint matrix to guide structure learning, ensuring inferred causal graphs adhere to predefined logical hierarchies and expert knowledge.

## Requirements and Dependencies

To run the notebook, you need a Python environment with the following libraries:

- **Python Standard Libraries**: `abc`, `itertools`, `json`, `logging`, `os`, `warnings`
- **Scientific Computing & Machine Learning**: `fsspec`, `matplotlib`, `networkx`, `numpy`, `pandas`, `pytorch_lightning`, `scipy`, `seaborn`, `scikit-learn`, `torch`
- **Causal Inference Frameworks**: `causica`, `causalnex`, `lingam`
- **Environment Management**: `dotenv` for loading environment variables

## Data

The notebook uses the **IBM Telco Customer Churn Dataset**. Required input files in the `data/` directory:

- `dataset.csv`: Primary dataset with customer information.
- `variables.json`: JSON file defining variable types and metadata for the DECI algorithm.

## Causal Discovery Algorithms

The pipeline implements the following algorithms, each trained with domain-specific constraints:

- **DECI (Differentiable Equilibrium-based Causal Inference)**: A deep learning-based probabilistic model approximating the posterior distribution over DAGs.
- **LiNGAM (Linear Non-Gaussian Acyclic Model)**: Identifies linear causal relationships and unique causal ordering for continuous, non-Gaussian data.
- **NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian)**: A gradient-based method learning a DAG by minimizing a penalized objective with a differentiable acyclicity constraint.
- **PC-GIN (PC with Generalized Independence with Noise)**: A constraint-based algorithm extending the PC algorithm with a residual-based independence test for mixed data types (commented out in the main pipeline).
- **GRaSP (Gradient-based Regularized Structure learning with Penalties)**: A score-based method learning a sparse DAG with custom acyclicity-constrained optimization (commented out in the main pipeline).

## How to Use

1. **Clone the Repository**: Ensure you have the project files, including `causation.ipynb` and the `data/` directory with `dataset.csv` and `variables.json`.
2. **Set up the Environment**: Install dependencies using a virtual environment.
3. **Run the Notebook**: Open `causation.ipynb` in a Jupyter Notebook environment (e.g., JupyterLab, VS Code with Python extension).
4. **Execute Cells Sequentially**: Run cells from top to bottom, structured into logical sections:

   - **Part A: Initialize the Project**: Loads libraries and configures global settings for reproducibility.
   - **Part B: Get the Data Prepared**: Handles data cleaning, outlier removal, feature encoding (binary and ordinal), and train/test splitting.
   - **Part C: Define the Functions**: Defines core functions, including `create_constraint_matrix`, `validate_constraints`, `save_relations_to_text`, `visualize_causal_graph`, and algorithm classes.
   - **Part D: Learn the Causal Graphs**: Executes the `run_causal_discovery_pipeline` function with defined tiered structures and constraints.

## Outputs

The pipeline creates a `causal_discovery_output/` directory with subdirectories for each algorithm (`DECI/`, `LiNGAM/`, `NOTEARS/`). Outputs include:

- **Causal Graphs (.png)**: Visualizations of discovered causal graphs with varying weight thresholds for sparsity.
  - Example: `deci_graph_thresh_0.30.png`, `lingam_graph_primary.png`, `notears_graph_additional_thresh_0.25.png`
- **Adjacency Matrices (.png)**: Heatmaps showing the strength of causal relationships.
  - Example: `deci_prob_matrix_heatmap.png`, `lingam_adj_matrix_heatmap.png`
- **Causal Relationships (.txt)**: Text files listing source, destination, and weight of causal edges for different thresholds.
  - Example: `deci_relations_primary.txt`, `lingam_relations_thresh_0.50.txt`, `notears_relations_additional_thresh_0.15.txt`
- **Logs and Summaries**: Detailed logs and a summary table showing edge count, graph density, and constraint violations for each algorithm.