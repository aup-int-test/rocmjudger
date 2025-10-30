import os
import random

# === Constraints ===
N_MIN, N_MAX = 1, 10000
C_MIN, C_MAX = 1, 1024
INPUT_MIN, INPUT_MAX = -100.0, 100.0
GAMMA_MIN, GAMMA_MAX = 0.1, 10.0
BETA_MIN, BETA_MAX = -10.0, 10.0
EPS = 1e-5
# ====================

os.makedirs("testcases", exist_ok=True)

def generate_case(path, N, C, seed):
    random.seed(seed)
    with open(path, "w") as f:
        # header
        f.write(f"{N} {C} {EPS}\n")

        # input matrix (N × C)
        for _ in range(N):
            line = " ".join(f"{random.uniform(INPUT_MIN, INPUT_MAX):.6f}" for _ in range(C))
            f.write(line + "\n")

        # gamma
        line = " ".join(f"{random.uniform(GAMMA_MIN, GAMMA_MAX):.6f}" for _ in range(C))
        f.write(line + "\n")

        # beta
        line = " ".join(f"{random.uniform(BETA_MIN, BETA_MAX):.6f}" for _ in range(C))
        f.write(line + "\n")

# ---- Normal random cases ----
for i, seed in enumerate([42, 123], start=1):
    N = random.randint(1, 64)
    C = random.randint(1, 32)
    generate_case(f"testcases/{i}.in", N, C, seed)

# ---- Edge case 1: minimal dimensions ----
generate_case("testcases/3.in", 1, 1, 9999)

# ---- Edge case 2: maximal dimensions ----
generate_case("testcases/4.in", 5000, 1024, 2025)

print("✅ Generated 4 BatchNorm testcases in ./testcases/")
