#!/usr/bin/env python3
"""
scripts/stat_utils.py

Statistical utility functions for proportion differences and confidence intervals (Checkpoint V).
Implements:
- Wilson score confidence interval for a single proportion.
- Newcombe hybrid-score confidence interval for the difference between two independent proportions (Method 10).
- Cell difference evaluation for paired Clean vs Hot comparisons with rep-to-rep stability checks.
"""

import math
from typing import Tuple, Dict, Any

def wilson_score_interval(x: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes Wilson score confidence interval for a single binomial proportion x/n.
    Reference: Wilson, E. B. (1927). JASA 22(158): 209-212.
    """
    if n <= 0:
        return 0.0, 0.0
    
    # Critical value z
    if abs(confidence - 0.95) < 1e-4:
        z = 1.959963984540054
    elif abs(confidence - 0.99) < 1e-4:
        z = 2.5758293035489004
    elif abs(confidence - 0.90) < 1e-4:
        z = 1.6448536269514722
    else:
        # Standard normal inverse approximation (Acklam or standard)
        z = 1.959963984540054

    p = x / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half_width = (z * math.sqrt(p * (1.0 - p) / n + (z ** 2) / (4.0 * (n ** 2)))) / denom
    
    l = max(0.0, center - half_width)
    u = min(1.0, center + half_width)
    return l, u

def newcombe_diff_ci(
    x1: int, n1: int, x2: int, n2: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes Newcombe hybrid-score confidence interval for the difference between two independent proportions (p1 - p2).
    Reference: Newcombe, R. G. (1998). "Interval estimation for the difference between independent proportions:
    comparison of eleven methods", Statistics in Medicine 17: 873-890 (Method 10).
    
    Returns:
        (diff, ci_lower, ci_upper) where diff = p1 - p2.
    """
    if n1 <= 0 or n2 <= 0:
        return 0.0, -1.0, 1.0

    p1 = x1 / n1
    p2 = x2 / n2
    diff = p1 - p2

    l1, u1 = wilson_score_interval(x1, n1, confidence=confidence)
    l2, u2 = wilson_score_interval(x2, n2, confidence=confidence)

    # Lower limit L
    d_low = math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    ci_lower = max(-1.0, diff - d_low)

    # Upper limit U
    d_high = math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    ci_upper = min(1.0, diff + d_high)

    return diff, ci_lower, ci_upper

def evaluate_cell_difference(
    clean_counts_rep0: int, clean_shots_rep0: int,
    clean_counts_rep1: int, clean_shots_rep1: int,
    hot_counts_rep0: int, hot_shots_rep0: int,
    hot_counts_rep1: int, hot_shots_rep1: int,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Evaluates CLEAN minus HOT difference pooled over 2 repetitions.
    Also computes rep0 vs rep1 stability checks.
    """
    # Repetition 0
    p_clean_r0 = clean_counts_rep0 / clean_shots_rep0 if clean_shots_rep0 > 0 else 0.0
    p_hot_r0 = hot_counts_rep0 / hot_shots_rep0 if hot_shots_rep0 > 0 else 0.0
    diff_r0 = p_clean_r0 - p_hot_r0

    # Repetition 1
    p_clean_r1 = clean_counts_rep1 / clean_shots_rep1 if clean_shots_rep1 > 0 else 0.0
    p_hot_r1 = hot_counts_rep1 / hot_shots_rep1 if hot_shots_rep1 > 0 else 0.0
    diff_r1 = p_clean_r1 - p_hot_r1

    # Stability checks (rep0 vs rep1 drift)
    clean_rep_drift = p_clean_r0 - p_clean_r1
    hot_rep_drift = p_hot_r0 - p_hot_r1

    # Pooled totals
    clean_counts_pooled = clean_counts_rep0 + clean_counts_rep1
    clean_shots_pooled = clean_shots_rep0 + clean_shots_rep1
    hot_counts_pooled = hot_counts_rep0 + hot_counts_rep1
    hot_shots_pooled = hot_shots_rep0 + hot_shots_rep1

    diff_pooled, ci_low, ci_high = newcombe_diff_ci(
        clean_counts_pooled, clean_shots_pooled,
        hot_counts_pooled, hot_shots_pooled,
        confidence=confidence
    )

    return {
        "p_clean_rep0": p_clean_r0,
        "p_clean_rep1": p_clean_r1,
        "p_hot_rep0": p_hot_r0,
        "p_hot_rep1": p_hot_r1,
        "diff_rep0": diff_r0,
        "diff_rep1": diff_r1,
        "clean_rep_drift": clean_rep_drift,
        "hot_rep_drift": hot_rep_drift,
        "p_clean_pooled": clean_counts_pooled / clean_shots_pooled if clean_shots_pooled > 0 else 0.0,
        "p_hot_pooled": hot_counts_pooled / hot_shots_pooled if hot_shots_pooled > 0 else 0.0,
        "diff_pooled": diff_pooled,
        "newcombe_ci_low": ci_low,
        "newcombe_ci_high": ci_high,
        "significant": (ci_low > 0.0 and diff_pooled >= 0.005) # >= 0.5 percentage points
    }

