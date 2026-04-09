from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from .settings import ExperimentConfig

# Base model configurations
BASE_MODELS = {
    'decision_tree': {
        'estimator': DecisionTreeClassifier,
        'params': {
            'random_state': ExperimentConfig.RANDOM_STATE
        }
    },
    'logistic_regression': {
        'estimator': LogisticRegression,
        'params': {
            'random_state': ExperimentConfig.RANDOM_STATE,
            'max_iter': 10000
        }
    }
}

# Hyperparameter search spaces (Random Search)
SEARCH_SPACES = {
    'decision_tree': {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': list(range(50, 400, 10)),
        'min_samples_leaf': list(range(20, 150, 10)),
        'max_leaf_nodes': list(range(250, 500, 15)),
        'max_depth': list(range(5, 30)),
    },
    'logistic_regression': {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [1, 5, 10, 30, 50, 80, 100, 125, 150, 200, 500],
        'max_iter': [1000],
    }
}


# Validation Curve Parameters
VAL_CURVE_PARAMS = {
    'decision_tree': {
        "max_depth": list(range(2, 10)) + list(range(10, 101, 10)),
        "min_samples_split": list(range(2, 25)) + list(range(25, 1001, 25)),
        "min_samples_leaf": list(range(1, 25)) + list(range(25, 1001, 25)),
        "max_leaf_nodes": list(range(2, 25)) + list(range(25, 1000, 25)),
    },
    'logistic_regression': {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'max_iter': [100, 200, 500, 1000, 10000]
    }
}