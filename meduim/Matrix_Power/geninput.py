import os
import random

# =====================================================
# Matrix Power input generator
# Constraints:
#   1 ≤ N ≤ 1024
#   1 ≤ P ≤ 20
#   -10.0 ≤ Aij ≤ 10.0
# =====================================================

os.makedirs("testcases", exist_ok=True)

def gen_case(filename, N, P):
    """Generate one test case file"""
    with open(filename, "w") as f:
        f.write(f"{N} {P}\n")
        for i in range(N):
            row = [f"{random.uniform(-10.0, 10.0):.4f}" for _ in range(N)]
            f.write(" ".join(row) + "\n")

# 1️⃣ Small random case
gen_case("testcases/1.in", N=4, P=3)

# 2️⃣ Medium case
gen_case("testcases/2.in", N=16, P=10)

# 3️⃣ Edge case: Identity (P=0 not allowed by constraint, so use P=1)
gen_case("testcases/3.in", N=32, P=1)

# 4️⃣ Large edge case: Maximum N with small P
gen_case("testcases/4.in", N=128, P=20)

print("✅ Generated 4 testcases in ./testcases/")
