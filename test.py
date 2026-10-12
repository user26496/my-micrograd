# test.py
from engine import Value


def check(label, actual, expected, tol=1e-5):
    ok = abs(actual - expected) < tol
    print(f"[{'✓' if ok else '✗'}] {label}: got {actual}, expected {expected}")


print("=== T1: Value + 数字 ===")
a = Value(2.0)
b = a + 3  # 期望 data=5
check("T1 data", b.data, 5.0)

print("\n=== T2: 数字 + Value（__radd__）===")
a = Value(2.0)
b = 3 + a
check("T2 data", b.data, 5.0)

print("\n=== T3: 取负 ===")
a = Value(2.0)
b = -a
b.backward()
check("T3 data", b.data, -2.0)
check("T3 grad", a.grad, -1.0)

print("\n=== T4: 减法 ===")
a = Value(5.0)
b = Value(3.0)
c = a - b
c.backward()
check("T4 data", c.data, 2.0)
check("T4 a.grad", a.grad, 1.0)
check("T4 b.grad", b.grad, -1.0)

print("\n=== T5: 幂 ===")
a = Value(3.0)
b = a**2
b.backward()
check("T5 data", b.data, 9.0)
check("T5 grad", a.grad, 6.0)  # ∂(a²)/∂a = 2a = 6

print("\n=== T6: 除法 ===")
a = Value(6.0)
b = Value(2.0)
c = a / b
c.backward()
check("T6 data", c.data, 3.0)
check("T6 a.grad", a.grad, 0.5)  # ∂(a/b)/∂a = 1/b = 0.5
check("T6 b.grad", b.grad, -1.5)  # ∂(a/b)/∂b = -a/b² = -6/4 = -1.5

print("\n=== T7: tanh ===")
import math

a = Value(0.5)
b = a.tanh()
b.backward()
expected_tanh = math.tanh(0.5)
expected_grad = 1 - expected_tanh**2
check("T7 data", b.data, expected_tanh)
check("T7 grad", a.grad, expected_grad)

print("\n=== T8: 组合 (a*2 + b).tanh() ===")
a = Value(0.3)
b = Value(0.5)
c = (a * 2 + b).tanh()
c.backward()
# 让 PyTorch 来确认（Phase 2.4）
print(f"  c.data = {c.data}")
print(f"  a.grad = {a.grad}")
print(f"  b.grad = {b.grad}")
