"""
LTI Engine v2.0 — One-Shot Causal Direction Discovery
Based on the final paper:
"Latent Transportable Interventions (LTI): One-Shot Causal Direction Discovery
via Differential Morphology and Phase-Space Hysteresis"

Key updates from v1:
- Replaced fixed δ=0.05 margin with Fisher-Z statistical significance test.
- Layer 2 (Hysteresis) is now bypassed directly for smooth (Q/Q) signals
  (not merely a tie-breaker).
- p-values are computed, making decisions sample-size-aware.
- Fully deterministic, training-free, O(T).
"""

import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------
# Helpers: Role Classification
# ----------------------------------------------------------------------
def classify_role(series, eps=1e-9):
    """
    Classify a 1D time-series as Effort (E), Flow (F), or Quantity (Q)
    based on derivative sparsity and zero-crossing morphology.
    """
    s = np.asarray(series, dtype=float)
    dx = np.gradient(s)

    # Sparsity = max(|dx|) / mean(|dx|)
    sparsity = np.max(np.abs(dx)) / (np.mean(np.abs(dx)) + eps)

    # Net change ratio (range in value space)
    net_change = abs(s[-1] - s[0]) / (np.max(s) - np.min(s) + eps)

    # Check for sign reversal (both positive and negative derivatives)
    has_sign_reversal = np.any(dx > 0) and np.any(dx < 0)

    if sparsity > 3.0 and has_sign_reversal and net_change < 0.3:
        return 'F'   # Flow: sparse, sign‑reversing, bounded
    elif sparsity > 3.0:
        return 'E'   # Effort: sparse, monotonic or pulsed
    else:
        return 'Q'   # Quantity: smooth, integrated signal

# ----------------------------------------------------------------------
# Layer 1: Asymmetric Derivative Correlation Score
# ----------------------------------------------------------------------
def causal_score(cause, effect, tau_max=12):
    """
    Compute C(A→B) = max_{τ} |corr( A(t), dB/dt(t+τ) )|
    """
    c = np.asarray(cause, dtype=float)
    de = np.gradient(np.asarray(effect, dtype=float))
    T = len(c)
    tau_max = min(tau_max, T - 2)

    best_r = 0.0
    best_n = 3  # minimum sample size
    for tau in range(0, tau_max + 1):
        if tau == 0:
            c_slice = c
            de_slice = de
        else:
            c_slice = c[:-tau]
            de_slice = de[tau:]

        n = len(c_slice)
        if n < 4:
            continue
        # Avoid numerical issues with near-constant signals
        if np.std(c_slice) < 1e-12 or np.std(de_slice) < 1e-12:
            continue

        r = np.corrcoef(c_slice, de_slice)[0, 1]
        if not np.isnan(r):
            abs_r = abs(r)
            if abs_r > best_r:
                best_r = abs_r
                best_n = n
    return best_r, best_n

# ----------------------------------------------------------------------
# Fisher Z-transform for significance of difference between two correlations
# ----------------------------------------------------------------------
def fisher_z_pvalue(r1, n1, r2, n2):
    """
    Two-sided p-value for H0: r1 == r2, using Fisher's Z transform.
    """
    # Clip to avoid numerical overflow in arctanh
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
    return p_value

# ----------------------------------------------------------------------
# Layer 2: Phase-Space Hysteresis (Shoelace formula)
# ----------------------------------------------------------------------
def hysteresis_area(cause, effect):
    """
    Compute directed area of the loop (X, Y) in phase space.
    Positive area ⇒ X → Y, Negative ⇒ Y → X
    """
    X = np.asarray(cause, dtype=float)
    Y = np.asarray(effect, dtype=float)

    # Min-max normalise
    X_norm = (X - np.min(X)) / (np.max(X) - np.min(X) + 1e-9)
    Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + 1e-9)

    # Mean-center to reduce boundary artifacts
    X_cent = X_norm - np.mean(X_norm)
    Y_cent = Y_norm - np.mean(Y_norm)

    # Shoelace formula (closed loop)
    area = 0.0
    for i in range(len(X_cent) - 1):
        area += X_cent[i] * Y_cent[i+1] - X_cent[i+1] * Y_cent[i]
    # Close the loop
    area += X_cent[-1] * Y_cent[0] - X_cent[0] * Y_cent[-1]

    return 0.5 * area

# ----------------------------------------------------------------------
# Main LTI Engine (public API)
# ----------------------------------------------------------------------
def robust_causal_direction(name_a, series_a, name_b, series_b, alpha=0.05):
    """
    Determine causal direction from a single bivariate time‑series.

    Parameters
    ----------
    name_a, name_b : str
        Labels for the two variables.
    series_a, series_b : array-like
        The observed time series (1D).
    alpha : float (default 0.05)
        Significance level for Fisher-Z test.

    Returns
    -------
    direction : str
        e.g. "A -> B" or "B -> A" or "UNDECIDABLE"
    confidence : float
        |A| (hysteresis area) or Δ (difference in correlations) or p-value.
    method : str
        "hyst" if decided by hysteresis, "deriv" if by Fisher-Z test.
    """
    # Convert to numpy arrays
    A = np.asarray(series_a, dtype=float)
    B = np.asarray(series_b, dtype=float)

    # 1. Classify roles
    role_a = classify_role(A)
    role_b = classify_role(B)

    # 2. If both are smooth (Q/Q), bypass Layer 1 entirely → Layer 2
    if role_a == 'Q' and role_b == 'Q':
        area = hysteresis_area(A, B)
        if area > 0:
            return f"{name_a} -> {name_b}", abs(area), "hyst"
        elif area < 0:
            return f"{name_b} -> {name_a}", abs(area), "hyst"
        else:
            return "UNDECIDABLE", 0.0, "hyst"

    # 3. Otherwise, compute derivative correlations
    r_ab, n_ab = causal_score(A, B)   # C(A→B)
    r_ba, n_ba = causal_score(B, A)   # C(B→A)

    # 4. Fisher-Z test for significance of difference
    p_val = fisher_z_pvalue(r_ab, n_ab, r_ba, n_ba)

    if p_val < alpha:
        # Statistically significant: decide by larger correlation
        if r_ab > r_ba:
            direction = f"{name_a} -> {name_b}"
        else:
            direction = f"{name_b} -> {name_a}"
        confidence = abs(r_ab - r_ba)
        method = "deriv"
        return direction, confidence, method
    else:
        # Tie → fallback to hysteresis (Layer 2)
        area = hysteresis_area(A, B)
        if area > 0:
            return f"{name_a} -> {name_b}", abs(area), "hyst"
        elif area < 0:
            return f"{name_b} -> {name_a}", abs(area), "hyst"
        else:
            return "UNDECIDABLE", 0.0, "hyst"
