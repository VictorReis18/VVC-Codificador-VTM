from .utils import log_message

# --- Grouping Logic Functions ---

def determine_size_group(row):
    w = row["W"]
    h = row["H"]
    max_dim = max(w, h)
    
    if max_dim == 128: return "128x128"
    elif max_dim == 64: return "64x64"
    elif max_dim == 32: return "32x32"
    elif max_dim == 16: return "16x16"
    elif max_dim == 8: return "8x8"
    else: return "4x4"

def determine_area_group(row):
    w = row["W"]
    h = row["H"]
    area = min(w, h) * max(w, h)
    
    area_to_group = {
        16: "G0", 32: "G1", 64: "G2", 128: "G3",
        256: "G4", 512: "G5", 1024: "G6", 2048: "G7",
        4096: "G8", 8192: "G9", 16384: "G10"
    }
    return area_to_group.get(area, "other")

def determine_all_group(row):
    w = row["W"]
    h = row["H"]
    return f"{w}x{h}"

def determine_orientation_group(row):
    w = row["W"]
    h = row["H"]
    if w == h: return "Square"
    elif w > h: return "Horizontal"
    else: return "Vertical"

def determine_aspect_ratio_group(row):
    w = row["W"]
    h = row["H"]
    ratio = max(w, h) / min(w, h)

    if abs(ratio - 1) < 0.01: return "1:1"
    elif abs(ratio - 2) < 0.01: return "2:1"
    elif abs(ratio - 4) < 0.01: return "4:1"
    elif abs(ratio - 8) < 0.01: return "8:1"
    elif abs(ratio - 16) < 0.01: return "16:1"
    elif abs(ratio - 32) < 0.01: return "32:1"
    else: return "other"

def determine_single_group(row):
    return "All_Blocks"

def determine_frame_level_group(row):
    level = row["FrameLevel"]
    
    if level == 0:
        return None
        
    return int(level)

# --- Strategy Map ---
GROUPING_STRATEGIES = {
    'area': determine_area_group,
    'max': determine_size_group,
    'orientation': determine_orientation_group,
    'aspect_ratio': determine_aspect_ratio_group,
    'all': determine_all_group,
    'single': determine_single_group,
    'frame_level': determine_frame_level_group
}

def apply_grouping_strategy(df, strategy_name):
    if strategy_name not in GROUPING_STRATEGIES:
        raise ValueError(f"Grouping strategy '{strategy_name}' not found.")
    
    log_message(f"Applying grouping strategy: {strategy_name}", level="INFO")
    
    df_out = df.copy()
    
    df_out["BlockGroup"] = df_out.apply(GROUPING_STRATEGIES[strategy_name], axis=1)
    
    before_drop = len(df_out)
    df_out = df_out.dropna(subset=["BlockGroup"])
    after_drop = len(df_out)
    
    if before_drop > after_drop:
        log_message(f"Strategy '{strategy_name}' filtered out {before_drop - after_drop} rows (e.g. FrameLevel 0).", level="WARNING")
    
    return df_out, sorted(df_out["BlockGroup"].unique())