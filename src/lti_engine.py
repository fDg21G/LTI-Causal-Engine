"""
Latent Transportable Interventions (LTI) Engine v5
Training-Data-Free One-Shot Causal Direction Discovery
"""
import numpy as np

def classify_role(series):
    s = np.array(series, dtype=float)
    dx = np.gradient(s)
    sparsity = np.max(np.abs(dx)) / (np.mean(np.abs(dx)) + 1e-9)
    net_change = abs(s[-1] - s[0]) / (np.max(s) - np.min(s) + 1e-9)
    has_sign_reversal = np.any(dx > 0) and np.any(dx < 0)
    
    if sparsity > 3.0 and has_sign_reversal and net_change < 0.3:
        return 'F'
    elif sparsity > 3.0:
        return 'E'
    else:
        return 'Q'

def causal_score(cause, effect):
    c = np.array(cause, dtype=float)
    de = np.gradient(np.array(effect, dtype=float))
    best = 0.0
    for lag in range(0, min(12, len(c)-2)):
        c_slice = c[:-lag] if lag > 0 else c
        de_slice = de[lag:] if lag > 0 else de
        if np.std(c_slice) < 1e-9 or np.std(de_slice) < 1e-9:
            continue
        r = np.corrcoef(c_slice, de_slice)[0, 1]
        if not np.isnan(r):
            best = max(best, abs(r))
    return best

def resolve_causal_tie(cause_candidate, effect_candidate):
    """Computes Phase-Space Hysteresis Area using the Shoelace formula."""
    X = np.array(cause_candidate, dtype=float)
    Y = np.array(effect_candidate, dtype=float)
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-9)
    Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-9)
    
    # FIX: Mean-centering to stabilize the orbit and avoid closing-edge artifacts
    X_val = X_norm - np.mean(X_norm)
    Y_val = Y_norm - np.mean(Y_norm)
    
    area = 0.0
    # Shoelace Formula for signed area
    for i in range(len(X_val) - 1):
        area += (X_val[i] * Y_val[i+1]) - (X_val[i+1] * Y_val[i])
        
    # FIX: Explicitly close the loop to prevent artificial boundaries
    area += (X_val[-1] * Y_val[0]) - (X_val[0] * Y_val[-1])
    
    return 0.5 * area

def robust_causal_direction(name_a, series_a, name_b, series_b):
    s_ab = causal_score(series_a, series_b)
    s_ba = causal_score(series_b, series_a)
    delta = abs(s_ab - s_ba)

    if delta < 0.05:
        area = resolve_causal_tie(series_a, series_b)
        if area > 0:
            return f"{name_a} -> {name_b}", abs(area), "Hysteresis Phase-Space"
        else:
            return f"{name_b} -> {name_a}", abs(area), "Hysteresis Phase-Space"
    else:
        if s_ab > s_ba:
            return f"{name_a} -> {name_b}", delta, "Derivative Correlation"
        else:
            return f"{name_b} -> {name_a}", delta, "Derivative Correlation"
