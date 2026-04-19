import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)


def simpson(N, a=0.0, b=24.0):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (y[0] + y[-1] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N:2])) * h / 3


I0 = simpson(10000)

Ns = np.arange(10, 1001, 2)
eps = np.abs([simpson(N) - I0 for N in Ns])
idx = np.where(eps <= 1e-12)[0]
N_opt = Ns[idx[0]] if len(idx) > 0 else 500

N0 = max(64, ((N_opt // 10) // 8) * 8)

I_N0 = simpson(N0)
I_half = simpson(N0 // 2)
I_R = I_N0 + (I_N0 - I_half) / 15

I1 = simpson(N0)
I2 = simpson(N0 // 2)
I3 = simpson(N0 // 4)
I_e = (I2 ** 2 - I1 * I3) / (2 * I2 - I1 - I3)
p = np.log(abs((I2 - I3) / (I1 - I2))) / np.log(2)

print(f"N_opt = {N_opt}, ε_opt = {eps[idx[0] if len(idx) > 0 else -1]:.2e}")
print(f"N0 = {N0}, ε0 = {abs(I_N0 - I0):.2e}")
print(f"RR:  I = {I_R:.10f}, ε = {abs(I_R - I0):.2e}")
print(f"E:   I = {I_e:.10f}, ε = {abs(I_e - I0):.2e}, p = {p:.2f}")

def adaptive_simpson(tol=1e-10, max_steps=1000):
    intervals = [(0.0, 24.0)]
    result = 0.0
    steps = 0

    while intervals and steps < max_steps:
        a, b = intervals.pop()
        h = b - a
        c = (a + b) / 2

        fa, fc, fb = f(a), f(c), f(b)
        S1 = h / 6 * (fa + 4 * fc + fb)

        d = (a + c) / 2
        e = (c + b) / 2
        fd, fe = f(d), f(e)
        S2 = h / 12 * (fa + 4 * fd + fc + 4 * fe + fb)

        if abs(S1 - S2) <= 15 * tol:
            result += S2
        else:
            intervals.append((a, c))
            intervals.append((c, b))
        steps += 1

    return result, steps * 5


print("\nАдаптивний алгоритм:")
for tol in [1e-6, 1e-8, 1e-10, 1e-12]:
    Iad, calls = adaptive_simpson(tol)
    print(f"δ = {tol:.0e}   ε = {abs(Iad - I0):.2e}   викликів ≈ {calls}")

# ====================== ГРАФІКИ ======================
x = np.linspace(0, 24, 1000)

plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', linewidth=2)
plt.title('Графік функції навантаження на сервер')
plt.xlabel('Час, x (години)')
plt.ylabel('Навантаження, f(x)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogy(Ns, eps, 'b-', linewidth=2, label='Похибка |I(N) - I₀|')
plt.axhline(1e-12, color='red', linestyle='--', label='Задана точність 1e-12')
plt.xlabel('Кількість підінтервалів N')
plt.ylabel('Похибка (логарифмічна шкала)')
plt.title('Залежність точності формули Сімпсона від N')
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()