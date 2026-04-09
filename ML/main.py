# ----------------------------------------------
from config.settings_decision_mts import DataConfig, ExperimentConfig, RESULTS_DIR, EXPORTS_DIR
#from config.settings_decision_intra import DataConfig, ExperimentConfig, RESULTS_DIR, EXPORTS_DIR
# ----------------------------------------------
   
import os
import sys
import shutil
from src.utils import log_message
from src.data import load_and_clean_data
from src.grouping import apply_grouping_strategy
from config.model_strategies import MODEL_STRATEGIES
from src.preprocessing import balance_group_data, split_and_sample, normalize_data, impute_data
from src.feature_selection import run_rfe
from src.training import tune_hyperparameters, train_final_model
from src.visualization import generate_validation_curves, generate_learning_curve
from src.evaluation import evaluate_and_save

def export_model_to_cpp(model, feature_names, class_names, function_name, output_dir, model_type):
    try:
        os.makedirs(output_dir, exist_ok=True)
        if model_type == 'decision_tree':
            import src.DecisionTreeToCpp as to_cpp
            to_cpp.save_code(model, feature_names, class_names, function_name=function_name)
            src_file = function_name + '.h'
            dst_file = os.path.join(output_dir, src_file)
            shutil.move(src_file, dst_file)
            log_message(f"✓ Tree exported: {dst_file}", level="INFO")
        elif model_type == 'logistic_regression':
            import src.LogisticRegToCpp as lr_to_cpp
            lr_to_cpp.save_code(model, feature_names, class_names, function_name=function_name, output_dir=output_dir)
            log_message(f"✓ LR exported: {output_dir}/{function_name}.h", level="INFO")
        else:
            log_message(f"Unknown model type for C++ export: {model_type}", level="ERROR")
    except ImportError:
        log_message(f"C++ export module not found. Export skipped.", level="ERROR")
    except Exception as e:
        log_message(f"Error exporting to C++: {e}", level="ERROR")


def main():
    log_message("=== Starting VVC ML Pipeline ===", level="stage")
    
    # 1. Load Data
    try:
        df_raw = load_and_clean_data(DataConfig.FILE_PATH)
        #df_raw = df_raw.sample(n=5000, random_state=42) # TODO: remover depois
    except Exception as e:
        log_message(f"Critical Error loading data: {e}", level="CRITICAL")
        sys.exit(1)

    # 2. Iterate over Grouping Strategies (area, max, single, etc.)
    for grouping_name in ExperimentConfig.ACTIVE_GROUPINGS:
        log_message(f"--- Grouping Strategy: {grouping_name} ---", level="stage")
        try:
            df_grouped, groups = apply_grouping_strategy(df_raw, grouping_name)
        except ValueError as e:
            log_message(f"Skipping strategy {grouping_name}: {e}", level="WARNING")
            continue

        # 3. Iterate over MODEL_STRATEGIES (Model A, Model B)
        for model_strategie_id, model_strategie_cfg in MODEL_STRATEGIES.items():
            log_message(f"--- model_strategie: {model_strategie_id} ({model_strategie_cfg['description']}) ---", level="stage")
            
            # Apply specific model_strategie transformation (label modification/filtering)
            df_model_strategie = model_strategie_cfg['process_function'](df_grouped)
            
            # Dynamic Model Type
            current_model_type = model_strategie_cfg['classifier_type']

            # 4. Iterate over Block Groups (e.g., 64x64, 32x32)
            for block_group in groups:
                group_id_clean = str(block_group).replace(":", "-").replace("×", "x")
                log_message(f"--- Block Group: {block_group} ---", level="stage")
                log_message(f"\n>>> Processing Group: {block_group} | Strategy: {grouping_name} | Model Type: {current_model_type}", level="INFO")
                
                # A. Filter by block group
                df_block = df_model_strategie[df_model_strategie['BlockGroup'] == block_group].copy()
                if df_block.empty:
                    continue
                
                # B. Balance Data
                df_balanced = balance_group_data(df_block)
                
                # Check for valid classes before splitting
                class_counts = df_balanced[DataConfig.TARGET_COLUMN].value_counts()
                if len(class_counts) < 2 or class_counts.min() < 2:
                    log_message(f"Skipping group {block_group}: not enough classes or samples per class for stratified splitting.", level="WARNING")
                    continue
                
                # C. Split Train/Test and Sample for Tuning
                try:
                    X_train, X_test, y_train, y_test, X_train_samp, y_train_samp = split_and_sample(df_balanced)
                except ValueError as e:
                    log_message(f"Skipping group {block_group} due to train/test split error: {e}", level="WARNING")
                    continue
                
                # C.0 Impute missing values if any
                X_train, X_test, X_train_samp = impute_data(X_train, X_test, X_train_samp)     
                               
                # C.1 Normalize Data if configured
                if ExperimentConfig.NORMALIZE_DATA:
                    X_train, X_test, X_train_samp = normalize_data(X_train, X_test, X_train_samp)
                
                # D. (Optional) Validation / Learning Curves
                if ExperimentConfig.RUN_VALIDATION_CURVES or ExperimentConfig.RUN_LEARNING_CURVES:
                    subdir = f"{grouping_name}_{model_strategie_id}_{group_id_clean}"
                    if ExperimentConfig.RUN_VALIDATION_CURVES:
                        generate_validation_curves(X_train_samp, y_train_samp, subdir, model_type=current_model_type)
                    if ExperimentConfig.RUN_LEARNING_CURVES:
                        generate_learning_curve(X_train_samp, y_train_samp, subdir, model_type=current_model_type, train_sizes=ExperimentConfig.LEARNING_CURVE_TRAIN_SIZES)
                    continue

                # E. Feature Selection (RFE) using dynamic model type
                log_message(f"--- Feature Selection (RFE) ---", level="stage")
                selected_cols = run_rfe(X_train_samp, y_train_samp, current_model_type)
                
                # F. Hyperparameter Tuning (Random Search)
                log_message(f"--- Hyperparameter Tuning ---", level="stage")
                best_params = tune_hyperparameters(X_train_samp[selected_cols], y_train_samp, current_model_type)
                
                # G. Final Training (Full Train set, Selected Features)
                log_message(f"--- Final Training ---", level="stage")
                final_model = train_final_model(X_train[selected_cols], y_train, current_model_type, best_params)
                
                # H. Evaluation (delegated to src.evaluation.evaluate_and_save)
                try:
                    evaluate_and_save(
                        final_model=final_model,
                        X_test=X_test,
                        y_test=y_test,
                        X_train=X_train,
                        y_train=y_train,
                        selected_cols=selected_cols,
                        grouping_name=grouping_name,
                        model_strategie_id=model_strategie_id,
                        block_group=block_group,
                        current_model_type=current_model_type,
                        best_params=best_params,
                        export_model_callback=export_model_to_cpp,
                    )
                except Exception as e:
                    log_message(f"Error during evaluation step: {e}", level="ERROR")

if __name__ == "__main__":
    main()