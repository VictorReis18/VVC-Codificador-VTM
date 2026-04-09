import os
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
EXPORTS_DIR = BASE_DIR / "cpp_exports"
LOG_FILE = BASE_DIR / "execution.log"

# --- Data Configuration ---
class DataConfig:
    FILE_PATH = DATA_DIR / "teste.csv"
    CSV_SEPARATOR = ','
    TARGET_COLUMN = 'Transformada'
    
    REMOVE_COLUMNS = []
    
    EXCLUDED_LINES = {}
    
    # Columns used for balancing logic
    BALANCE_COLUMNS = ['Transformada', 'QP', 'W', 'H']

# --- Experiment Configuration ---
class ExperimentConfig:
    RANDOM_STATE = 42
    N_JOBS = -1
    TEST_SIZE = 0.25
    MAX_SAMPLES_PER_CLASS = 200000
    NORMALIZE_DATA = True
    
    # Handling Missing Values
    # --> True: Impute missing values
    # --> False: Remove any row with missing values (drop)
    IMPUTE_MISSING_VALUES = False
    
    # Cross Validation
    CV_FOLDS = 5
    SCORING = 'accuracy'
    
    # Feature Selection (RFCV)
    RFE_ENABLED = True
    RFE_STEP = 1
    RFE_MIN_FEATURES = 5
    
    # Hyperparameter Tuning
    RANDOM_SEARCH_ITER = 2000
    
    # Flags
    RUN_VALIDATION_CURVES = False
    RUN_LEARNING_CURVES = False
    RUN_LEARNING_CURVES_AT_END = True
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
    EXPORT_CPP = True
    
    # Active Grouping Strategies
    # Options: 'area', 'max', 'orientation', 'aspect_ratio', 'all', 'single'
    ACTIVE_GROUPINGS = ['all']
