import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 
from config.settings import DataConfig, ExperimentConfig
from .utils import log_message

def balance_group_data(df_group):
    """Balances data within a specific block group."""
    target_col = DataConfig.TARGET_COLUMN
    
    if df_group[target_col].nunique() < 2:
        log_message(f"Group has less than 2 classes. Skipping balance.", level="WARNING")
        return df_group

    # Create composite key for balancing
    # Only use columns that exist in the dataframe
    valid_cols = [c for c in DataConfig.BALANCE_COLUMNS if c in df_group.columns]
    missing_cols = [c for c in DataConfig.BALANCE_COLUMNS if c not in df_group.columns]
    if missing_cols:
        log_message(f"Missing balance columns: {missing_cols}", level="WARNING")
    
    if not valid_cols:
        log_message(f"No valid columns found for balancing. Skipping balance.", level="ERROR")
        return df_group

    df_group['balance_key'] = list(zip(*(df_group[c] for c in valid_cols)))
    
    min_samples = df_group['balance_key'].value_counts().min()
    
    if min_samples < 1:
         return df_group.drop(columns=['balance_key'])

    balanced_df = pd.concat([
        resample(g, replace=False, n_samples=min_samples, random_state=ExperimentConfig.RANDOM_STATE)
        for _, g in df_group.groupby('balance_key')
    ])
    
    log_message(f"Group balanced. Total: {len(balanced_df)} samples.", level="INFO")
    return balanced_df.drop(columns=['balance_key'])


def normalize_data(X_train, X_test, X_train_samp):
    """Normalizes data using StandardScaler (fit on Train only)."""
    log_message("Normalizing data (fit on Train only)...", level="INFO")
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    feature_names = X_train.columns.tolist()
    
    X_train_norm = pd.DataFrame(scaler.transform(X_train), columns=feature_names, index=X_train.index)
    X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)
    X_train_samp_norm = pd.DataFrame(scaler.transform(X_train_samp), columns=feature_names, index=X_train_samp.index)
    
    means_str = ", ".join(f"{m:.6f}" for m in scaler.mean_)
    scales_str = ", ".join(f"{s:.6f}" for s in scaler.scale_) # scale_ é o desvio padrão
    
    log_message(f"Normalized params:", level="DEBUG")
    log_message(f"- Feature Order: {feature_names}", level="DEBUG")
    log_message(f"- means[] = {{ {means_str} }};", level="DEBUG")
    log_message(f"- scales[] = {{ {scales_str} }};", level="DEBUG")
    
    return X_train_norm, X_test_norm, X_train_samp_norm


def impute_data(X_train, X_test, X_train_samp):
    """Imputes missing values using a hybrid strategy: Mean for floats, Mode for ints/objects."""
    log_message("Imputing missing values ...", level="INFO")
    
    float_cols = X_train.select_dtypes(include=['float', 'float32', 'float64']).columns # Floats -> Average
    cat_cols = X_train.select_dtypes(include=['int', 'int32', 'int64', 'object', 'category']).columns  # Ints/Objects -> Mode
    
    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    X_train_samp_imp = X_train_samp.copy()
    
    impute_map = {}
    
    if len(float_cols) > 0:
        imputer_mean = SimpleImputer(strategy='mean')
        imputer_mean.fit(X_train[float_cols])
        
        X_train_imp[float_cols] = imputer_mean.transform(X_train[float_cols])
        X_test_imp[float_cols] = imputer_mean.transform(X_test[float_cols])
        X_train_samp_imp[float_cols] = imputer_mean.transform(X_train_samp[float_cols])
        
        for col, val in zip(float_cols, imputer_mean.statistics_):
            impute_map[col] = val

    if len(cat_cols) > 0:
        imputer_mode = SimpleImputer(strategy='most_frequent')
        imputer_mode.fit(X_train[cat_cols])
        
        X_train_imp[cat_cols] = imputer_mode.transform(X_train[cat_cols])
        X_test_imp[cat_cols] = imputer_mode.transform(X_test[cat_cols])
        X_train_samp_imp[cat_cols] = imputer_mode.transform(X_train_samp[cat_cols])
        
        for col, val in zip(cat_cols, imputer_mode.statistics_):
            impute_map[col] = val
            
    feature_names = X_train.columns.tolist()
    ordered_vals = [impute_map[col] for col in feature_names]
    
    vals_str = ", ".join(f"{v:.6f}" for v in ordered_vals)
    
    log_message("=== Imputation Parameters -  Mean (floats) / Mode (ints) ===", level="INFO")
    log_message(f"- Feature Order: {feature_names}", level="DEBUG")
    log_message(f"- impute_vals[] = {{ {vals_str} }};", level="DEBUG")
    
    if getattr(ExperimentConfig, 'IMPUTE_MISSING_VALUES', False):
        log_message(">> Applying imputation to datasets (Config=True).", level="WARNING")
        return X_train_imp, X_test_imp, X_train_samp_imp
    else:
        log_message(">> Skipping imputation application (Config=False). Using original data.", level="INFO")
        return X_train, X_test, X_train_samp


def split_and_sample(df):
    """Splits into Train/Test and creates a smaller sample for Tuning."""
    X = df.drop(columns=[DataConfig.TARGET_COLUMN, 'BlockGroup'], errors='ignore')
    y = df[DataConfig.TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=ExperimentConfig.TEST_SIZE, 
        random_state=ExperimentConfig.RANDOM_STATE, 
        stratify=y
    )
    
    # Sampling for faster Hyperparameter Search
    train_data = pd.concat([X_train, y_train], axis=1)
    train_sampled = pd.concat([
        resample(g, replace=False, 
                 n_samples=min(len(g), ExperimentConfig.MAX_SAMPLES_PER_CLASS),
                 random_state=ExperimentConfig.RANDOM_STATE)
        for _, g in train_data.groupby(DataConfig.TARGET_COLUMN)
    ])
    
    X_train_samp = train_sampled.drop(columns=[DataConfig.TARGET_COLUMN])
    y_train_samp = train_sampled[DataConfig.TARGET_COLUMN]
    
    return X_train, X_test, y_train, y_test, X_train_samp, y_train_samp