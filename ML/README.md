# 🚀 ML Training Pipeline

This project implements a complete **Machine Learning pipeline** designed to automate the workflow and optimize decision-making processes. It features dynamic block grouping, recursive feature selection (RFE), hyperparameter optimization, and robust preprocessing.

---

## 📂 Key Features

- **End-to-End Pipeline:** Automated loading, cleaning, preprocessing, training, and validation.
- **Dynamic Grouping Strategies:** Allows you to configure data partitions before training to create specialized models: such as block size, area, or orientation.
- **Automated Feature Selection:** Utilizes `RFECV` (Recursive Feature Elimination with Cross-Validation) to identify the optimal feature subset per group.
- **Hyperparameter Tuning:** Integrated `RandomizedSearchCV` for Decision Trees and Logistic Regression.
- **C++ Export Engine:** Automatically converts trained `sklearn` models into optimized C++ header files (`.h`) with inline `if-else` logic.
- **Robust Preprocessing:** Imputation strategy and intelligent class balancing.

---

## 🧠 Workflow Architecture

The pipeline follows a modular structure executed via `main.py`:

```
A[Load Data] --> B[Grouping Strategy]
B --> C[Preprocessing & Balancing]
C --> D[Feature Selection (RFE)]
D --> E[Hyperparameter Tuning]
E --> F[Final Training]
F --> G[Export to C++]
```

---

## ⚙️ Configuration & Strategies

The pipeline is highly configurable via the config/ files. Below are the strategies supported by the codebase.

### 1. Grouping Strategies (`src/grouping.py`)

The pipeline splits the dataset into subsets to train specialized models.

| Strategy     | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| max          | Groups by the maximum block dimension (e.g., 64x64, 32x32). |
| area         | Groups by total block area (G0 to G10).                     |
| orientation  | Splits into: Square, Horizontal, Vertical.                  |
| aspect_ratio | Based on aspect ratio (1:1, 2:1, 4:1, etc.).                |
| single       | Treats all blocks as a single group (All_Blocks).           |

### 2. Supported Models (`model_strategies.py`)

| Model Type          | Implementation Details                                |
| ------------------- | ----------------------------------------------------- |
| decision_tree       | Optimized for if/else rules.                          |
| logistic_regression | Linear classification.                                |

---

## 📦 Project Structure

```bash
ml-training-pipeline/
├── config/
│   ├── settings_decision_intra.py  # Config for Intra decisions
│   ├── settings_decision_mts.py    # Config for MTS decisions
│   └── model_hyperparameters.py    # Search spaces for Tuning
├── src/
│   ├── data.py                 # Data loading and cleaning
│   ├── grouping.py             # Block partitioning logic
│   ├── preprocessing.py        # Imputation, Normalization, Split
│   ├── feature_selection.py    # RFE (Recursive Feature Elimination)
│   ├── training.py             # Training loops and parameter search
│   └── DecisionTreeToCpp.py    # Python -> C++ Converter
├── main.py                     # Entry point
└── results/                    # Reports and Learning Curves
```

---

## 🚀 How to Run

1. **Setup Environment**

   - Ensure you have the required dependencies installed (`requeriments.txt`).

2. **Configure Experiment**
   - Edit and import the desired `config/settings.py` file (e.g., `import settings_decision_intra.py`):

```python
class ExperimentConfig:
    RFE_ENABLED = True           # Enable feature selection
    IMPUTE_MISSING_VALUES = True # Handle nulls automatically (Mean/Mode)
    EXPORT_CPP = True            # Generate .h files at the end
    ACTIVE_GROUPINGS = ['max']   # Grouping strategy
```

3. **Execute**
   - Run the main script:

```bash
python3 main.py
```

## 📊 Logging & Visualization

- **Files:** Complete log saved to `execution.log`.
- **Classification Reports:** After each model is trained and evaluated, the pipeline logs the full accuracy report (precision, recall, f1-score, support). These reports are also saved as text files in the `results/` directory for later analysis.
- **Curves:** Generates validation and learning curves in `results/learning_curves/`.
