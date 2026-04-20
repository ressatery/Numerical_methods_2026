import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (8, 5)
})

os.makedirs("lab4", exist_ok=True)

def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def dM_numeric(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

t0 = 1
exact = dM_exact(t0)

print(f"Точне значення M'(1) = {exact:.6f}")

eps_machine = np.finfo(float).eps

d3M = lambda t: -0.001 * np.exp(-0.1*t) - 5*np.sin(t)
C = abs(d3M(t0)) / 6

h_opt_theory = (eps_machine / C) ** (1/3)
print(f"Теоретичний оптимальний крок: h_opt = {h_opt_theory:.8f}")

h_theory = np.logspace(-14, 0, 500)
approx_component = C * h_theory**2
round_component  = 2 * eps_machine / h_theory
total_theory     = approx_component + round_component

h_measured = np.logspace(-8, 0, 60)
errors_measured = [abs(dM_numeric(t0, h) - exact) for h in h_measured]

optimal_h = h_measured[np.argmin(errors_measured)]

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(h_measured, errors_measured, '-o', color='#2563EB', linewidth=2, markersize=3, label="Похибка")
ax.axvline(x=optimal_h, color='gray', linestyle='--', linewidth=1, label=f"h_opt = {optimal_h:.5f}")

ax.set_xlabel("Крок h")
ax.set_ylabel("Похибка")
ax.set_title("Залежність похибки від кроку h")
ax.legend(fontsize=10)
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.7)

plt.tight_layout()
plt.savefig("lab4/plot_error.png", dpi=300)
plt.close()

print(f"Оптимальний крок (виміри):    h_opt = {optimal_h:.8f}  ")

h = 0.01
D_h  = dM_numeric(t0, h)
D_2h = dM_numeric(t0, h / 2)
D_4h = dM_numeric(t0, h / 4)

p = 2
D_rr = D_2h + (D_2h - D_h) / (2**p - 1)

numerator   = (D_2h ** 2) - D_4h * D_h
denominator = 2 * D_2h - (D_4h + D_h)
D_aitken = numerator / denominator if abs(denominator) > 1e-15 else D_h


methods = ["Точне", "D(h)", "D(h/2)", "Рунге-Ромберг", "Ейткен"]
values  = [exact,   D_h,    D_2h,     D_rr,             D_aitken]

plt.figure()
plt.plot(methods, values, marker='o', label="Значення")
plt.axhline(y=exact, linestyle='--', color='red', label="Точне значення")
plt.title("Порівняння методів обчислення похідної")
plt.ylabel("Значення похідної")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig("lab4/plot_methods.png", dpi=300)
plt.close()

t_vals = np.linspace(0, 2, 100)
M_vals = M(t_vals)
tangent = exact * (t_vals - t0) + M(t0)

plt.figure()
plt.plot(t_vals, M_vals, label="M(t)")
plt.plot(t_vals, tangent, linestyle="--", label="Дотична в t=1")
plt.scatter([t0], [M(t0)], zorder=5)
plt.title("Функція та її похідна (геометрично)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.savefig("lab4/plot_function.png", dpi=300)
plt.close()

print("\n" + "="*58)
print("  РЕЗУЛЬТАТИ  (t₀ = 1,  h = 0.01)")
print("="*58)
print(f"  Точне значення   : {exact:.8f}")
print(f"  D(h)             : {D_h:.8f}    похибка {abs(D_h     - exact):.8f}")
print(f"  D(h/2)           : {D_2h:.8f}    похибка {abs(D_2h    - exact):.8f}")
print(f"  Рунге-Ромберг    : {D_rr:.8f}    похибка {abs(D_rr    - exact):.10f}")
print(f"  Ейткен           : {D_aitken:.8f}    похибка {abs(D_aitken - exact):.10f}")
print("="*58)