from config.settings import DataConfig

def _process_model_A(df):
    """
    Model A:
    - No filtering.
    - Labels: (0 or 1) -> 10, others -> 20.
    """
    df_out = df.copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x in [0, 1] else 20
    )
    return df_out

def _process_model_B(df):
    """
    Model B:
    - Remove rows where Target is 0 or 1.
    - Labels: (2) -> 10, others -> 20.
    """
    df_out = df[~df[DataConfig.TARGET_COLUMN].isin([0, 1])].copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x == 2 else 20
    )
    return df_out

def _process_raw(df):
    """No processing, return raw dataframe."""
    return df.copy()

def _process_dst7_dst7(df):
    """
    Model dst7_dst7
    - Label 10 if target == 2
    - Otherwise, 20
    """
    df_out = df.copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x == 2 else 20
    )
    return df_out

def _process_dct8_dst7(df):
    """
    Model dct8_dst7
    - Label 10 if target == 3
    - Otherwise, 20
    """
    df_out = df.copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x == 3 else 20
    )
    return df_out

def _process_dst7_dct8(df):
    """
    Model dst7_dct8
    - Label 10 if target == 4
    - Otherwise, 20
    """
    df_out = df.copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x == 4 else 20
    )
    return df_out

def _process_dct8_dct8(df):
    """
    Model dct8_dct8
    - Label 10 if target == 5
    - Otherwise, 20
    """
    df_out = df.copy()
    df_out[DataConfig.TARGET_COLUMN] = df_out[DataConfig.TARGET_COLUMN].apply(
        lambda x: 10 if x == 5 else 20
    )
    return df_out



# Scenario Definitions including dynamic classifier type
MODEL_STRATEGIES = {
    "decision_tree": {
        "description": "Decision Tree",
        "process_function": _process_raw,
        "classifier_type": "decision_tree"
    }
}

"""
MODEL_STRATEGIES = {
    "A": {
        "description": "Model A (Filter after DCT2)",
        "process_function": _process_model_A,
        "classifier_type": "decision_tree"
    },
    "B": {
        "description": "Model B (DST7 vs Others)",
        "process_function": _process_model_B,
        "classifier_type": "decision_tree"
    }
}
"""

"""

MODEL_STRATEGIES = {
    "dst7_dst7": {
        "description": "Modelo DST7 (target=2 -> 10)",
        "process_function": _process_dst7_dst7,
        "classifier_type": "decision_tree"
    },
    "dct8_dst7": {
        "description": "Modelo DCT8_DST7 (target=3 -> 10)",
        "process_function": _process_dct8_dst7,
        "classifier_type": "decision_tree"
    },
    "dst7_dct8": {
        "description": "Modelo DST7_DCT8 (target=4 -> 10)",
        "process_function": _process_dst7_dct8,
        "classifier_type": "decision_tree"
    },
    "dct8_dct8": {
        "description": "Modelo DCT8_DCT8 (target=5 -> 10)",
        "process_function": _process_dct8_dct8,
        "classifier_type": "decision_tree"
    }
}
"""