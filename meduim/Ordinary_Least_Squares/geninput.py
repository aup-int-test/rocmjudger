import os
import random

os.makedirs("testcases", exist_ok=True)

def generate_case(path, seed):
    random.seed(seed)
    n_samples = random.randint(1, 100000)
    n_features = random.randint(1, min(1000, n_samples))

    with open(path, "w") as f:
        f.write(f"{n_samples} {n_features}\n")
        for _ in range(n_samples * n_features):
            f.write(f"{random.uniform(-1000.0, 1000.0):.6f}\n")
        for _ in range(n_samples):
            f.write(f"{random.uniform(-1000.0, 1000.0):.6f}\n")

for i, seed in enumerate([42, 123, 999, 2025], start=1):
    path = f"testcases/{i}.in"
    generate_case(path, seed)

print("✅ Generated 4 testcases in ./testcases/")
