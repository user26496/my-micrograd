from engine import Value
import torch
import math


def check(label, actual, expected, tol=1e-5):
    ok = abs(actual - expected) < tol
    marker = "✓" if ok else "✗"
    print(
        f"[{marker}] {label}: micrograd={actual:.6f}, pytorch={expected:.6f}, diff={abs(actual-expected):.2e}"
    )


def to_torch(x):
    """把 Python 数值包装成 PyTorch 的 leaf tensor（requires_grad=True）"""
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


# ========================================
# Test 1: 线性表达式 L = (a*b + c) * f
# ========================================
print("=" * 60)
print("Test 1: 线性表达式 L = (a*b + c) * f")
print("=" * 60)

# micrograd
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)
L = (a * b + c) * f
L.backward()

# PyTorch
at, bt, ct, ft = to_torch(2.0), to_torch(-3.0), to_torch(10.0), to_torch(-2.0)
Lt = (at * bt + ct) * ft
Lt.backward()

check("a.grad", a.grad, at.grad.item())
check("b.grad", b.grad, bt.grad.item())
check("c.grad", c.grad, ct.grad.item())
check("f.grad", f.grad, ft.grad.item())
check("L.data", L.data, Lt.item())


# ========================================
# Test 2: 非线性 tanh
# ========================================
print("\n" + "=" * 60)
print("Test 2: 单神经元 output = tanh(w*x + b)")
print("=" * 60)

x = Value(0.5)
w = Value(1.5)
b = Value(-0.3)
output = (w * x + b).tanh()
output.backward()

xt, wt, bt = to_torch(0.5), to_torch(1.5), to_torch(-0.3)
output_t = torch.tanh(wt * xt + bt)
output_t.backward()

check("x.grad", x.grad, xt.grad.item())
check("w.grad", w.grad, wt.grad.item())
check("b.grad", b.grad, bt.grad.item())
check("output.data", output.data, output_t.item())


# ========================================
# Test 3: 复用变量（验证 += 关键）
# ========================================
print("\n" + "=" * 60)
print("Test 3: 复用变量 L = a*b + a*a + a")
print("=" * 60)

a = Value(3.0)
b = Value(4.0)
L = a * b + a * a + a
L.backward()

at, bt = to_torch(3.0), to_torch(4.0)
Lt = at * bt + at * at + at
Lt.backward()

check("a.grad", a.grad, at.grad.item())  # 应该是 2a+b+1 = 11
check("b.grad", b.grad, bt.grad.item())  # 应该是 a = 3
check("L.data", L.data, Lt.item())


# ========================================
# Test 4: 混合数字 + 除法 + 幂
# ========================================
print("\n" + "=" * 60)
print("Test 4: 复杂表达式 L = (a + 2) * (b**3) / c - 1")
print("=" * 60)

a = Value(1.5)
b = Value(2.0)
c = Value(4.0)
L = (a + 2) * (b**3) / c - 1
L.backward()

at, bt, ct = to_torch(1.5), to_torch(2.0), to_torch(4.0)
Lt = (at + 2) * (bt**3) / ct - 1
Lt.backward()

check("a.grad", a.grad, at.grad.item())
check("b.grad", b.grad, bt.grad.item())
check("c.grad", c.grad, ct.grad.item())
check("L.data", L.data, Lt.item())


# ========================================
# Test 5: Karpathy 视频里的那个"神经元"
# ========================================
print("\n" + "=" * 60)
print("Test 5: Karpathy 视频的 micrograd 招牌示例")
print("=" * 60)

# 两个输入 + 两个权重 + 一个偏置 + tanh = 一个神经元
x1 = Value(2.0)
x2 = Value(0.0)
w1 = Value(-3.0)
w2 = Value(1.0)
b = Value(6.8813735870195432)
n = x1 * w1 + x2 * w2 + b
o = n.tanh()
o.backward()

x1t, x2t = to_torch(2.0), to_torch(0.0)
w1t, w2t = to_torch(-3.0), to_torch(1.0)
bt = to_torch(6.8813735870195432)
nt = x1t * w1t + x2t * w2t + bt
ot = torch.tanh(nt)
ot.backward()

check("x1.grad", x1.grad, x1t.grad.item())
check("x2.grad", x2.grad, x2t.grad.item())
check("w1.grad", w1.grad, w1t.grad.item())
check("w2.grad", w2.grad, w2t.grad.item())
check("b.grad", b.grad, bt.grad.item())
check("o.data", o.data, ot.item())

print("\n" + "=" * 60)
print("如果以上全部 ✓，恭喜：你已经成功复现了 micrograd 🏆")
print("=" * 60)
