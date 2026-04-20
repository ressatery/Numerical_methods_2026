import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8')
plt.rcParams.update({"font.size": 12, "figure.figsize": (9, 5)})
os.makedirs("lab5", exist_ok=True)

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12) ** 2)

print("=" * 55)

def simpson(N, a=0.0, b=24.0):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (y[0] + y[-1] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N:2])) * h / 3

I0 = simpson(10000)

print(f"  Точне значення інтегралу I₀ = {I0:.8f}")
print("=" * 55)

Ns  = np.arange(10, 1001, 2)
eps = np.array([abs(simpson(N) - I0) for N in Ns])

TARGET = 1e-12
idx   = np.where(eps <= TARGET)[0]
N_opt = int(Ns[idx[0]]) if len(idx) > 0 else int(Ns[np.argmin(eps)])
eps_opt = abs(simpson(N_opt) - I0)

print(f"\n  N_opt = {N_opt}   ε_opt = {eps_opt:.10f}")

N0 = 6
I_N0 = simpson(N0)
eps0 = abs(I_N0 - I0)

print(f"  N0   = {N0}   ε0   = {eps0:.8f}")

p_rr  = 4
I_2N0 = simpson(2 * N0)
I_R   = I_2N0 + (I_2N0 - I_N0) / (2 ** p_rr - 1)
eps_R = abs(I_R - I0)

print(f"\n  Рунге-Ромберг:")
print(f"    I(N0)    = {I_N0:.8f}   (N={N0})")
print(f"    I(2·N0)  = {I_2N0:.8f}   (N={2*N0})")
print(f"    I_RR     = {I_R:.8f}   ε = {eps_R:.10f}")

I1 = I_N0
I2 = I_2N0
I3 = simpson(4 * N0)

numerator   = I2 ** 2 - I1 * I3
denominator = 2 * I2 - I1 - I3
I_E   = numerator / denominator if abs(denominator) > 1e-15 else I2
eps_E = abs(I_E - I0)

p_est = np.log(abs((I1 - I2) / (I2 - I3))) / np.log(2) if abs(I2 - I3) > 1e-15 else 0

print(f"\n  Ейткен:")
print(f"    I(N0)    = {I1:.8f}   (N={N0})")
print(f"    I(2·N0)  = {I2:.8f}   (N={2*N0})")
print(f"    I(4·N0)  = {I3:.8f}   (N={4*N0})")
print(f"    I_E      = {I_E:.8f}   ε = {eps_E:.10f}")
print(f"    Оцінка порядку p ≈ {p_est:.2f}")

def adaptive_simpson(tol=1e-10, a=0.0, b=24.0):
    calls = [0]

    def S(a, b):
        c = (a + b) / 2
        calls[0] += 3
        return (b - a) / 6 * (f(a) + 4 * f(c) + f(b))

    def recurse(a, b, tol, S_ab):
        c = (a + b) / 2
        S_ac = S(a, c)
        S_cb = S(c, b)
        if abs(S_ac + S_cb - S_ab) <= 15 * tol:
            return S_ac + S_cb
        return recurse(a, c, tol/2, S_ac) + recurse(c, b, tol/2, S_cb)

    result = recurse(a, b, tol, S(a, b))
    return result, calls[0]

print(f"\n  Адаптивний алгоритм (Сімпсон):")
print(f"  {'δ':>10}   {'ε':>12}   {'викликів':>10}")
print(f"  {'-'*38}")
for tol in [1e-4, 1e-6, 1e-8, 1e-10]:
    Iad, calls = adaptive_simpson(tol)
    print(f"  {tol:10.0e}   {abs(Iad - I0):12.6e}   {calls:>10}")

print("=" * 55)

x_plot = np.linspace(0, 24, 1000)
plt.figure()
plt.plot(x_plot, f(x_plot), color='#2563EB', linewidth=2)
plt.fill_between(x_plot, f(x_plot), alpha=0.12, color='#2563EB')
plt.title("Функція навантаження на сервер f(x)")
plt.xlabel("Час, x (години)")
plt.ylabel("Навантаження, f(x)")
plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
plt.tight_layout()
plt.savefig("lab5/plot_function.png", dpi=300)
plt.close()

plt.figure()
plt.semilogy(Ns, eps, color='#2563EB', linewidth=2, label="Похибка |I(N) − I₀|")
plt.axhline(TARGET, color='#DC2626', linestyle='--', linewidth=1.2,
            label=f"Задана точність {TARGET:.0e}")
plt.axvline(N_opt,  color='gray',    linestyle=':',  linewidth=1.2,
            label=f"N_opt = {N_opt}")
plt.xlabel("Кількість підінтервалів N")
plt.ylabel("Похибка")
plt.title("Залежність точності формули Сімпсона від N")
plt.legend(fontsize=10)
plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)
plt.tight_layout()
plt.savefig("lab5/plot_error.png", dpi=300)
plt.close()

methods = ["Simpson\nN0", "Simpson\nN0/2", "Рунге-\nРомберг", "Ейткен"]
errors  = [eps0, abs(I2 - I0), eps_R, eps_E]
colors  = ['#60A5FA', '#34D399', '#F59E0B', '#A78BFA']

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, errors, color=colors, edgecolor='white', linewidth=0.8)
plt.yscale('log')
for bar, e in zip(bars, errors):
    plt.text(bar.get_x() + bar.get_width() / 2, e * 1.5,
             f"{e:.1e}", ha='center', va='bottom', fontsize=10)
plt.ylabel("Похибка (лог. шкала)")
plt.title("Порівняння точності методів інтегрування")
plt.grid(True, axis='y', linestyle="--", linewidth=0.4, alpha=0.7)
plt.tight_layout()
plt.savefig("lab5/plot_methods.png", dpi=300)
plt.close()

print("\n  Графіки збережено у папку lab5/")