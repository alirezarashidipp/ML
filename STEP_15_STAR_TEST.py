import math

def compute_star(clarity, role, goal, value, ac_present):
    clarity_clamped = min(max(clarity, 5), 95)
    x_c = (clarity_clamped - 5) / 90.0
    x_s = (role + goal + value) / 3.0
    x_a = 1.0 if ac_present else 0.0

    K = ((3 * x_s) + (2 * x_a) + 1) / 6.0
    alpha = 0.7
    beta = 0.3
    I = (x_c ** alpha) * (K ** beta)
    Y_cont = 1 + 4 * I

    if x_a == 0:
        Y_capped = min(Y_cont, 3.0)
    else:
        Y_capped = Y_cont

    Y_final = round(2 * Y_capped) / 2.0
    return Y_final


samples = [
    (10, 0, 0, 0, 0),
    (30, 1, 0, 0, 0),
    (60, 1, 1, 0, 1),
    (80, 1, 1, 1, 0),
    (90, 1, 1, 1, 1),
]

for i, (clarity, role, goal, value, ac) in enumerate(samples, start=1):
    star = compute_star(clarity, role, goal, value, ac)
    print(f"Sample {i}: clarity={clarity}, RGV=({role},{goal},{value}), AC={ac} -> stars = {star}")
