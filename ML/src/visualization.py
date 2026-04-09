import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from config.settings import ExperimentConfig, RESULTS_DIR
from config.model_hyperparameters import VAL_CURVE_PARAMS, BASE_MODELS
from .utils import log_message

def generate_validation_curves(X, y, output_subdir, model_type):
    log_message(f"Generating validation curves for {output_subdir} (model: {model_type})...")

    # Instantiate estimator from BASE_MODELS
    if model_type not in BASE_MODELS:
        raise ValueError(f"Unknown model_type for validation curves: {model_type}")

    model_conf = BASE_MODELS[model_type]
    clf = model_conf['estimator'](**model_conf.get('params', {}))

    save_path = RESULTS_DIR / "val_curves" / output_subdir
    os.makedirs(save_path, exist_ok=True)

    # Resolve parameter ranges: explicit val_curve_params -> VAL_CURVE_PARAMS -> derive from SEARCH_SPACES
    param_map = {}
    if model_type in VAL_CURVE_PARAMS:
        param_map = VAL_CURVE_PARAMS[model_type]
    else:
        log_message(f"No validation curve parameters defined for model '{model_type}'. Skipping validation curves.")
        return

    if not param_map:
        log_message(f"No numeric parameter ranges found for model '{model_type}'. Skipping validation curves.")
        return

    for param_name, param_range in param_map.items():
        try:
            train_scores, test_scores = validation_curve(
                clf, X, y,
                param_name=param_name,
                param_range=param_range,
                scoring=ExperimentConfig.SCORING,
                cv=ExperimentConfig.CV_FOLDS,
                n_jobs=ExperimentConfig.N_JOBS
            )
        except Exception as e:
            log_message(f"Skipping param '{param_name}' for model '{model_type}': {e}")
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, np.mean(train_scores, axis=1), label="Training", marker='o')
        plt.plot(param_range, np.mean(test_scores, axis=1), label="Validation (CV)", marker='o')
        plt.title(f"Validation Curve - {model_type} :: {param_name}")
        plt.xlabel(param_name)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path / f"{param_name}.png")
        plt.close()

    log_message(f"Validation curves saved to {save_path}")




def generate_learning_curve(X, y, output_subdir, model_type, train_sizes=np.linspace(0.1, 1.0, 10), best_params=None):
    """
    Generate learning curve plots for a given model and dataset.

    If `best_params` is provided, they will override the default estimator
    parameters defined in `BASE_MODELS` when instantiating the estimator.
    """

    log_message(f"Generating learning curve for {output_subdir} (model: {model_type})...")

    # Validate model
    if model_type not in BASE_MODELS:
        raise ValueError(f"Unknown model_type for learning curves: {model_type}")

    model_conf = BASE_MODELS[model_type]
    # Merge default params with any best_params provided (best_params takes precedence)
    base_params = dict(model_conf.get('params', {}))
    if best_params:
        merged_params = base_params.copy()
        merged_params.update(best_params)
    else:
        merged_params = base_params

    clf = model_conf['estimator'](**merged_params)

    # Output directory
    save_path = RESULTS_DIR / "learning_curves" / output_subdir
    os.makedirs(save_path, exist_ok=True)

    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator=clf,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=ExperimentConfig.CV_FOLDS,
            scoring=ExperimentConfig.SCORING,
            n_jobs=ExperimentConfig.N_JOBS,
            shuffle=True,
            random_state=ExperimentConfig.RANDOM_STATE
        )
    except Exception as e:
        log_message(f"Error while generating learning curve for model '{model_type}': {e}")
        return

    # Compute means and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, label="Training Score", marker='o')
    plt.plot(train_sizes_abs, test_mean, label="Validation (CV) Score", marker='o')

    # Error bands (optional but useful)
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.2)

    plt.title(f"Learning Curve - {model_type}")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path / "learning_curve.png")
    plt.close()

    log_message(f"Learning curve saved to {save_path}")
