import os
import random

os.makedirs("testcases", exist_ok=True)

def generate_case(path, seed, edge=False):
    random.seed(seed)

    # edge cases: smallest and largest sizes
    if edge and seed == 42:
        n_samples = 1
        n_features = 1
    elif edge and seed == 123:
        n_samples = 100000
        n_features = 1000
    else:
        n_samples = random.randint(10, 50000)
        n_features = random.randint(1, min(1000, n_samples))

    with open(path, "w") as f:
        f.write(f"{n_samples} {n_features}\n")

        # X matrix
        for _ in range(n_samples * n_features):
            val = random.uniform(-10.0, 10.0)
            f.write(f"{val:.6f}\n")

        # y vector (binary 0/1)
        for _ in range(n_samples):
            y_val = random.randint(0, 1)
            f.write(f"{y_val}\n")

for i, seed in enumerate([42, 123, 999, 2025], start=1):
    path = f"testcases/{i}.in"
    edge = (i <= 2)  # first two are edge cases
    generate_case(path, seed, edge)

print("✅ Generated 4 logistic regression testcases in ./testcases/")
