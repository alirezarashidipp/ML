import math

# Define configuration constants outside the function for easy tuning
ALPHA = 0.7
BETA = 0.3
PENALTY_CAP = 3.0
CLARITY_MIN = 5
CLARITY_MAX = 95
STAR_MIN = 1.0

def compute_star_robust(clarity: float, role: int, goal: int, value: int, ac_present: int) -> float:
    """
    Computes a ticket's quality star rating (1.0 to 5.0, in 0.5 steps)
    using the Geometric Interaction model with a penalty for missing AC.
    """
    # 1. Input Validation (Essential for robustness)
    for flag, name in zip([role, goal, value, ac_present], ['role', 'goal', 'value', 'ac_present']):
        if flag not in (0, 1):
            raise ValueError(f"Input flag '{name}' must be 0 or 1. Received: {flag}")
    
    if not isinstance(clarity, (int, float)):
         raise TypeError(f"Clarity score must be numeric. Received: {type(clarity)}")

    # 2. Normalization and Clamping
    clarity_clamped = min(max(clarity, CLARITY_MIN), CLARITY_MAX)
    x_c = (clarity_clamped - CLARITY_MIN) / (CLARITY_MAX - CLARITY_MIN)
    
    x_s = (role + goal + value) / 3.0
    x_a = float(ac_present)

    # 3. Geometric Interaction Core (MSc-level logic)
    # K: Completeness Multiplier (Smoothed)
    K = ((3 * x_s) + (2 * x_a) + 1) / 6.0
    
    # I: Combined Quality Index (Geometric Mean)
    I = math.pow(x_c, ALPHA) * math.pow(K, BETA)
    
    # Y_cont: Scale to 1-5 range
    Y_cont = STAR_MIN + 4.0 * I

    # 4. Penalty Application (Hard Gate for missing AC)
    if ac_present == 0:
        Y_capped = min(Y_cont, PENALTY_CAP)
    else:
        Y_capped = Y_cont

    # 5. Final Rounding
    # Ensures the output snaps to the nearest 0.5 step
    Y_final = round(2 * Y_capped) / 2.0
    
    # Final check to ensure min star rating is 1.0 (since the math can sometimes result in slightly less than 1)
    return max(STAR_MIN, Y_final)

# --- Example Usage ---
# The logic remains the same, but the function call uses the robust version
samples = [
    (10, 0, 0, 0, 0),
    (30, 1, 0, 0, 0),
    (60, 1, 1, 0, 1),
    (80, 1, 1, 1, 0),
    (90, 1, 1, 1, 1),
]

print("--- Robust Output ---")
for i, (clarity, role, goal, value, ac) in enumerate(samples, start=1):
    star = compute_star_robust(clarity, role, goal, value, ac)
    print(f"Sample {i}: clarity={clarity}, RGV=({role},{goal},{value}), AC={ac} -> stars = {star}")
