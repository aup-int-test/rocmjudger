import sys
import random

a = 0
b = 2
n = 100000000

print(f"{a} {b} {n}")

for i in range(n):
    x = a + (b - a) * i / (n - 1)
    y = x * x
    print(f"{y:.6f}", end=' ')
print()
