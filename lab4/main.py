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

print("Точне значення:", round(exact, 6))

h_values = np.logspace(-4, -1, 50)
errors = []

for h in h_values:
    approx = dM_numeric(t0, h)
    errors.append(abs(approx - exact))

plt.figure()
plt.loglog(h_values, errors, marker='o', label="Похибка")
plt.xlabel("Крок h (лог шкала)")
plt.ylabel("Похибка (лог шкала)")
plt.title("Залежність похибки від кроку")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.savefig("lab4/plot_error.png", dpi=300)
plt.close()

optimal_h = h_values[np.argmin(errors)]
print("Оптимальний крок:", optimal_h)

h = 0.01
D_h = dM_numeric(t0, h)
D_h2 = dM_numeric(t0, h/2)
D_h4 = dM_numeric(t0, h/4)

p = 2
D_rr = D_h2 + (D_h2 - D_h) / (2**p - 1)

D_aitken = D_h - ((D_h2 - D_h)**2) / (D_h4 - 2*D_h2 + D_h)

methods = ["Exact", "h", "h/2", "Рунге", "Ейткен"]
values = [exact, D_h, D_h2, D_rr, D_aitken]

plt.figure()
plt.plot(methods, values, marker='o')
plt.axhline(y=exact, linestyle='--', label="Точне значення")

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
plt.scatter([t0], [M(t0)])

plt.title("Функція та її похідна (геометрично)")
plt.xlabel("t")
plt.ylabel("M(t)")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)

plt.savefig("lab4/plot_function.png", dpi=300)
plt.close()

print("\n--- Результати ---")
print("D(h):", round(D_h, 6))
print("D(h/2):", round(D_h2, 6))
print("Рунге:", round(D_rr, 6))
print("Ейткен:", round(D_aitken, 6))