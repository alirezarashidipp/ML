from star_model import compute_star_robust


samples = [
    (10, 0, 0, 0, 0),
    (30, 1, 0, 0, 0),
    (60, 1, 1, 0, 1),
    (80, 1, 1, 1, 0),
    (90, 1, 1, 1, 1),
    (95, 0, 1, 1, 1),
    (5,  1, 1, 1, 1),
]

print("--- Manual Tests for compute_star_robust ---")
for i, (clarity, role, goal, value, ac) in enumerate(samples, start=1):
    star = compute_star_robust(clarity, role, goal, value, ac)
    print(f"Test {i}: clarity={clarity}, RGV=({role},{goal},{value}), AC={ac} -> stars = {star}")
