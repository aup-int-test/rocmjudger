import os
import argparse
import random
import math

def f(x):
    # 可調整你想積分的函數，例如 x^2
    return x * x * x

def write_testcase(filename, a, b, y_samples):
    n_samples = len(y_samples)
    os.makedirs("testcases", exist_ok=True)
    with open(os.path.join("testcases", filename), "w") as f_out:
        f_out.write(f"{a} {b} {n_samples}\n")
        f_out.write(" ".join(f"{y:.6f}" for y in y_samples) + "\n")
    print(f"✅ 測資寫入 testcases/{filename}")

def generate_manual_case(args):
    a, b, n = args.a, args.b, args.n
    if n > 1:
        x_samples = [a + (b - a) * i / (n - 1) for i in range(n)]
    elif n == 1:
        x_samples = [(a + b) / 2]
    else:
        x_samples = []

    y_samples = [f(x) for x in x_samples]
    write_testcase(args.output, a, b, y_samples)

def generate_preset_case(preset):
    cases = {
        "zero-length":   lambda: (2, 2, [f(2)] * 10000000),
        "n0":            lambda: (0, 1, []),
        "n1":            lambda: (0, 1, [f(0.5)]),
        "negative-range":lambda: (-2, 2, [f(x) for x in [-2 + 4 * i / 7 for i in range(8)]]),
        "large-n":       lambda: (0, 1, [f(random.uniform(0, 1)) for _ in range(1000)]),
        "constant-y":    lambda: (0, 1, [1.0 for _ in range(10)]),
        "tiny-values":   lambda: (0, 1, [1e-30 for _ in range(10000000)]),
        "huge-values":   lambda: (0, 1, [1e+30 for _ in range(10000000)]),
        "non-integer-ab":lambda: (0.1, 1.9, [f(x) for x in [0.1 + 1.8 * i / 9 for i in range(10)]]),
        "random-y":      lambda: (0, 1, [random.uniform(-1, 1) for _ in range(100000000)]),
    }

    if preset not in cases:
        print(f"❌ Unknown preset: {preset}")
        return

    a, b, y_samples = cases[preset]()
    filename = f"{preset}.txt"
    write_testcase(filename, a, b, y_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test cases for Monte Carlo integration.")
    parser.add_argument("--a", type=float, help="Lower bound of integration (a)")
    parser.add_argument("--b", type=float, help="Upper bound of integration (b)")
    parser.add_argument("--n", type=int, help="Number of samples")
    parser.add_argument("--output", type=str, default="input.txt", help="Output filename (inside testcases/)")
    parser.add_argument("--preset", type=str, help="Generate predefined edge case")

    args = parser.parse_args()

    if args.preset:
        generate_preset_case(args.preset)
    elif args.a is not None and args.b is not None and args.n is not None:
        generate_manual_case(args)
    else:
        print("❗請提供 --a --b --n 參數，或使用 --preset 模式產生測資")
