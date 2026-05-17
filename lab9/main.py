import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8')
plt.rcParams.update({"font.size": 11, "figure.figsize": (9, 6)})
os.makedirs("lab9", exist_ok=True)

def hooke_jeeves(F, x0, h0=0.5, alpha=2.0, eps1=1e-8, eps2=1e-8,
                 step_reduce=0.5, max_iter=50000):
    n = len(x0)
    x_base = np.array(x0, dtype=float)
    h = np.full(n, h0, dtype=float)
    trajectory = [x_base.copy()]

    def exploratory_search(x_start, h_vec):
        x = x_start.copy()
        for i in range(n):
            x_plus = x.copy();  x_plus[i] += h_vec[i]
            if F(x_plus) < F(x):
                x = x_plus
            else:
                x_minus = x.copy();  x_minus[i] -= h_vec[i]
                if F(x_minus) < F(x):
                    x = x_minus
        return x

    for iteration in range(max_iter):
        x_new = exploratory_search(x_base, h)

        if F(x_new) < F(x_base):
            while True:
                x_pattern = x_new + alpha * (x_new - x_base)
                trajectory.append(x_new.copy())

                x_exp = exploratory_search(x_pattern, h)

                if F(x_exp) < F(x_new):
                    x_base = x_new.copy()
                    x_new  = x_exp.copy()
                else:
                    x_base = x_new.copy()
                    break
        else:
            h *= step_reduce

            if np.max(h) < eps1 and F(x_base) < eps2:
                trajectory.append(x_base.copy())
                return x_base, iteration + 1, F(x_base), trajectory
            if np.max(h) < eps1 * 1e-4:
                trajectory.append(x_base.copy())
                return x_base, iteration + 1, F(x_base), trajectory

    trajectory.append(x_base.copy())
    return x_base, max_iter, F(x_base), trajectory

def rosenbrock(x):
    """Функція Розенброка"""
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def power_func(x):
    """Степенева"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def wood(x):
    """Функція Вуда"""
    return (100*(x[1]-x[0]**2)**2 + (1-x[0])**2
          + 90*(x[3]-x[2]**2)**2 + (1-x[2])**2
          + 10.1*((x[1]-1)**2 + (x[3]-1)**2)
          + 19.8*(x[1]-1)*(x[3]-1))


print("=" * 60)
print("  ТЕСТУВАННЯ МЕТОДУ ХУКА-ДЖИВСА")
print("=" * 60)

tests = [
    ("Розенброка",  rosenbrock,  [-1.2, 1.0],         "(1, 1)"),
    ("Степенева",  power_func,  [0.0,  0.0],         "(3, 2)"),
    ("Вуда (4D)",   wood,        [-3.0,-1.0,-3.0,-1.0],"(1,1,1,1)"),
]

test_results = {}
for name, func, x0, known_min in tests:
    xm, it, fm, tr = hooke_jeeves(func, x0, h0=0.5, eps1=1e-9, eps2=1e-14)
    test_results[name] = (xm, it, fm, tr)
    coords = ", ".join(f"{v:.6f}" for v in xm)
    print(f"\n  [{name}]  x₀ = {x0}")
    print(f"    x*    = ({coords})")
    print(f"    F(x*) = {fm:.2e}   кроків = {it}")
    print(f"    Точний мінімум: {known_min}")

def f1(xy): return xy[0]**2 + xy[1]**2 - 4
def f2(xy): return xy[0]**2 - xy[1]   - 1
def target(xy): return f1(xy)**2 + f2(xy)**2

x2_exact = (1 + np.sqrt(13)) / 2
x_exact  =  np.sqrt(x2_exact)
y_exact  =  x2_exact - 1

print(f"\n{'=' * 67}")
print("  СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ")
print("  f₁: x² + y² = 4   (коло)")
print("  f₂: x² − y  = 1   (парабола)")
print(f"  Аналітичний розв'язок:  x* = ±{x_exact:.8f},  y* = {y_exact:.8f}")
print(f"  Перевірка: f₁={f1([x_exact,y_exact]):.1e},  f₂={f2([x_exact,y_exact]):.1e}")
print("=" * 67)

starts = [(1.0, 1.0), (-1.0, 1.0)]
solutions = []

for x0_sys in starts:
    xm, it, fm, traj = hooke_jeeves(
        target, list(x0_sys), h0=0.5, eps1=1e-10, eps2=1e-18, max_iter=100000
    )
    solutions.append((xm, it, fm, traj))
    print(f"\n  Початкове наближення: x₀ = {x0_sys}")
    print(f"  x* = ({xm[0]:.10f}, {xm[1]:.10f})")
    print(f"  Φ(x*) = {fm:.2e}   кроків по траєкторії = {it}")
    print(f"  f₁(x*) = {f1(xm):.2e},   f₂(x*) = {f2(xm):.2e}")

for k, (xm, it, fm, traj) in enumerate(solutions):
    fname = f"lab9/trajectory_{k+1}.txt"
    with open(fname, "w", encoding="utf-8") as fp:
        fp.write(f"# Траєкторія спуску, початок {starts[k]}\n")
        fp.write(f"{'крок':>6}  {'x':>14}  {'y':>14}  {'Phi(x,y)':>14}\n")
        fp.write("-" * 56 + "\n")
        for i, pt in enumerate(traj):
            fp.write(f"{i:6d}  {pt[0]:14.8f}  {pt[1]:14.8f}  {target(pt):14.6e}\n")
    print(f"\n  Збережено: {fname}  ({len(traj)} точок траєкторії)")


TRAJ_COLORS = ['#7C3AED', '#EA580C']

# Графік 1: криві рівнянь + траєкторії plot_system
fig, ax = plt.subplots(figsize=(8, 7))

theta = np.linspace(0, 2*np.pi, 500)
ax.plot(2*np.cos(theta), 2*np.sin(theta), 'b-', linewidth=2.2, label='$f_1: x^2+y^2=4$')

xp = np.linspace(-2.2, 2.2, 500)
ax.plot(xp, xp**2 - 1, 'r-', linewidth=2.2, label='$f_2: x^2-y=1$')

# Точки перетину
for sign in [1, -1]:
    ax.scatter([sign*x_exact], [y_exact], color='#16A34A', s=130, zorder=6)
    ax.annotate(f'({sign*x_exact:.4f}, {y_exact:.4f})',
                xy=(sign*x_exact, y_exact),
                xytext=(sign*x_exact + 0.18*sign, y_exact - 0.3),
                fontsize=9, color='#166534')

for k, (xm, it, fm, traj) in enumerate(solutions):
    traj_arr = np.array(traj)
    ax.plot(traj_arr[:,0], traj_arr[:,1], '-o',
            color=TRAJ_COLORS[k], markersize=3, linewidth=1.3,
            label=f'Траєкторія {k+1} (x₀={starts[k]}, {len(traj)} кроків)')
    ax.scatter([traj[0][0]], [traj[0][1]], color=TRAJ_COLORS[k],
               s=90, marker='s', zorder=7)

ax.set_xlim(-2.8, 2.8); ax.set_ylim(-1.8, 3.5)
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Система рівнянь та траєкторії методу Хука-Дживса')
ax.legend(fontsize=9.5); ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)
plt.tight_layout()
plt.savefig('lab9/plot_system.png', dpi=300)
plt.close()

#Графік 2: контурний + траєкторії plot_contour
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for k, (ax, (xm, it, fm, traj)) in enumerate(zip(axes, solutions)):
    xg = np.linspace(-2.5, 2.5, 350)
    yg = np.linspace(-1.5, 3.2, 350)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = f1([Xg, Yg])**2 + f2([Xg, Yg])**2

    levels = np.logspace(-4, 2, 28)
    cp = ax.contourf(Xg, Yg, Zg, levels=levels, cmap='Blues_r', alpha=0.88)
    ax.contour(Xg, Yg, Zg, levels=levels, colors='white', linewidths=0.35, alpha=0.35)
    plt.colorbar(cp, ax=ax, shrink=0.85, label='Φ(x,y)')

    traj_arr = np.array(traj)
    ax.plot(traj_arr[:,0], traj_arr[:,1], 'o-',
            color='#F59E0B', markersize=2.5, linewidth=1.3, label='Траєкторія')
    ax.scatter([traj[0][0]], [traj[0][1]], color='white', s=80,
               marker='s', zorder=5, label=f'Старт {starts[k]}')
    ax.scatter([xm[0]], [xm[1]], color='#DC2626', s=120,
               marker='*', zorder=5, label=f'x*=({xm[0]:.4f}, {xm[1]:.4f})')

    ax.set_title(f'Φ(x,y) — контурний графік, x₀={starts[k]}')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)

plt.suptitle('Цільова функція Φ = f₁² + f₂² та траєкторії спуску', y=1.02)
plt.tight_layout()
plt.savefig('lab9/plot_contour.png', dpi=300, bbox_inches='tight')
plt.close()

#Графік 3: збіжність Φ по кроках plot_convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for k, (ax, (xm, it, fm, traj)) in enumerate(zip(axes, solutions)):
    fvals = [max(target(pt), 1e-20) for pt in traj]
    ax.semilogy(range(len(fvals)), fvals, color=TRAJ_COLORS[k], linewidth=2)
    ax.axhline(1e-16, color='gray', linestyle='--', linewidth=1,
               label='Ціль 10⁻¹⁶')
    ax.set_title(f'Збіжність Φ(x,y), x₀={starts[k]}')
    ax.set_xlabel('Крок траєкторії'); ax.set_ylabel('Φ(x,y)')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)

plt.suptitle('Зменшення цільової функції по кроках', y=1.02)
plt.tight_layout()
plt.savefig('lab9/plot_convergence.png', dpi=300, bbox_inches='tight')
plt.close()

#Графік 4: функція Розенброка plot_rosenbrock
xm_ros, _, _, traj_ros = test_results["Розенброка"]

fig, ax = plt.subplots(figsize=(8, 6))
xg = np.linspace(-2.0, 1.8, 350)
yg = np.linspace(-0.5, 2.0, 350)
Xg, Yg = np.meshgrid(xg, yg)
Zg = 100*(Yg - Xg**2)**2 + (1 - Xg)**2

levels = np.logspace(-1, 3.5, 30)
ax.contourf(Xg, Yg, Zg, levels=levels, cmap='YlOrRd_r', alpha=0.82)
ax.contour(Xg, Yg, Zg, levels=levels, colors='white', linewidths=0.3, alpha=0.3)

traj_arr = np.array(traj_ros)
ax.plot(traj_arr[:,0], traj_arr[:,1], 'o-',
        color='#2563EB', markersize=2.5, linewidth=1.2, label='Траєкторія')
ax.scatter([-1.2], [1.0], color='white', s=80, marker='s', zorder=5,
           label='Старт (−1.2, 1.0)')
ax.scatter([xm_ros[0]], [xm_ros[1]], color='#16A34A', s=120,
           marker='*', zorder=5, label=f'x*=({xm_ros[0]:.4f}, {xm_ros[1]:.4f})')
ax.scatter([1], [1], color='red', s=80, marker='x',
           linewidths=2, zorder=6, label='Точний мінімум (1, 1)')

ax.set_title('Функція Розенброка — тест методу Хука-Дживса')
ax.set_xlabel('x₁'); ax.set_ylabel('x₂')
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.savefig('lab9/plot_rosenbrock.png', dpi=300)
plt.close()