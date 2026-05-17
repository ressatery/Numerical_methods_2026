import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8')
plt.rcParams.update({"font.size": 11, "figure.figsize": (9, 5)})
os.makedirs("lab8", exist_ok=True)

TOL = 1e-10
MAX_ITER = 1000

def f(x):
    return x * np.exp(x) - 3 * np.sin(x) - 1

def df(x):
    return np.exp(x) + x * np.exp(x) - 3 * np.cos(x)

def ddf(x):
    return 2 * np.exp(x) + x * np.exp(x) + 3 * np.sin(x)

A, B = -2.0, 4.0

# Крок 1: Табуляція, пошук наближених коренів
step = 0.1
xs = np.arange(A, B + step, step)
ys = f(xs)

with open("lab8/tabulation.txt", "w", encoding="utf-8") as fout:
    fout.write(f"{'x':>8}  {'f(x)':>14}\n")
    fout.write("-" * 26 + "\n")
    for xi, yi in zip(xs, ys):
        fout.write(f"{xi:8.3f}  {yi:14.6f}\n")

# Знайти наближені корені (зміна знаку)
roots_approx = []
for i in range(len(ys) - 1):
    if ys[i] * ys[i + 1] < 0:
        mid = (xs[i] + xs[i + 1]) / 2
        roots_approx.append(mid)

print("=" * 67)
print("  ТРАНСЦЕНДЕНТНЕ РІВНЯННЯ: f(x) = x·eˣ − 3·sin(x) − 1")
print("=" * 67)
print(f"\n  Наближені корені (зміна знаку):")
for i, r in enumerate(roots_approx):
    print(f"    x{i+1} ≈ {r:.3f}   f = {f(r):.4f}")

# два корені f' > 0 — зростання, f' < 0 — спадання)
roots_growing  = [r for r in roots_approx if df(r) > 0]
roots_falling  = [r for r in roots_approx if df(r) < 0]

x0_rise = roots_growing[0]  if roots_growing else roots_approx[0]
x0_fall = roots_falling[-1] if roots_falling else roots_approx[-1]

print(f"\n  Обрані початкові наближення:")
print(f"    Зростання: x₀ = {x0_rise:.3f}   f'(x₀) = {df(x0_rise):.4f}")
print(f"    Спадання:  x₀ = {x0_fall:.3f}   f'(x₀) = {df(x0_fall):.4f}")

def converged(x_new, x_old):
    return abs(x_new - x_old) < TOL and abs(f(x_new)) < TOL

def simple_iteration(x0, lam=None):
    if lam is None:
        lam = -1.0 / df(x0)
    x = x0
    history = [x]
    for _ in range(MAX_ITER):
        x_new = x + lam * f(x)
        history.append(x_new)
        if converged(x_new, x):
            break
        x = x_new
    return x_new, len(history) - 1, history


def newton(x0):
    x = x0
    history = [x]
    for _ in range(MAX_ITER):
        fx, dfx = f(x), df(x)
        if abs(dfx) < 1e-14:
            break
        x_new = x - fx / dfx
        history.append(x_new)
        if converged(x_new, x):
            break
        x = x_new
    return x_new, len(history) - 1, history


def chebyshev(x0):
    """Метод Чебишева (третій порядок)"""
    x = x0
    history = [x]
    for _ in range(MAX_ITER):
        fx, dfx, ddfx = f(x), df(x), ddf(x)
        if abs(dfx) < 1e-14:
            break
        x_new = x - fx / dfx - ddfx * fx**2 / (2 * dfx**3)
        history.append(x_new)
        if converged(x_new, x):
            break
        x = x_new
    return x_new, len(history) - 1, history


def chord(x0, x1=None):
    """Метод хорд:"""
    if x1 is None:
        x1 = x0 + 0.1 * np.sign(df(x0)) if df(x0) != 0 else x0 + 0.1
    history = [x0, x1]
    x_prev, x_cur = x0, x1
    for _ in range(MAX_ITER):
        fx_cur  = f(x_cur)
        fx_prev = f(x_prev)
        denom = fx_cur - fx_prev
        if abs(denom) < 1e-14:
            break
        x_new = x_cur - fx_cur * (x_cur - x_prev) / denom
        history.append(x_new)
        if converged(x_new, x_cur):
            break
        x_prev, x_cur = x_cur, x_new
    return x_new, len(history) - 1, history


def parabola(x0, x1=None, x2=None):
    h = 0.1
    if x1 is None:
        x1 = x0 + h
    if x2 is None:
        x2 = x0 + 2 * h
    history = [x0, x1, x2]
    xa, xb, xc = x0, x1, x2
    for _ in range(MAX_ITER):
        fa, fb, fc = f(xa), f(xb), f(xc)
        h1 = xb - xa
        h2 = xc - xb
        d1 = (fb - fa) / h1
        d2 = (fc - fb) / h2
        A2 = (d2 - d1) / (h1 + h2)
        B2 = d2 + h2 * A2
        C2 = fc
        disc = B2**2 - 4 * A2 * C2
        if abs(A2) < 1e-14:
            if abs(B2) < 1e-14:
                break
            delta = -C2 / B2
        else:
            sq = np.sqrt(complex(disc))
            d_plus  = -2 * C2 / (B2 + sq)
            d_minus = -2 * C2 / (B2 - sq)
            delta = d_plus.real if abs(d_plus) <= abs(d_minus) else d_minus.real
        x_new = xc + delta
        history.append(x_new)
        if converged(x_new, xc):
            break
        xa, xb, xc = xb, xc, x_new
    return x_new, len(history) - 1, history


def reverse_interpolation(x0, x1=None, x2=None):
    """Метод зворотної інтерполяції (інтерполяція Лагранжа по вузлах"""
    h = 0.1
    if x1 is None:
        x1 = x0 + h
    if x2 is None:
        x2 = x0 + 2 * h
    history = [x0, x1, x2]
    xa, xb, xc = x0, x1, x2
    for _ in range(MAX_ITER):
        fa, fb, fc = f(xa), f(xb), f(xc)
        denom_a = (fa - fb) * (fa - fc)
        denom_b = (fb - fa) * (fb - fc)
        denom_c = (fc - fa) * (fc - fb)
        if abs(denom_a) < 1e-14 or abs(denom_b) < 1e-14 or abs(denom_c) < 1e-14:
            break
        x_new = (xa * fb * fc / denom_a
               + xb * fa * fc / denom_b
               + xc * fa * fb / denom_c)
        history.append(x_new)
        if converged(x_new, xc):
            break
        xa, xb, xc = xb, xc, x_new
    return x_new, len(history) - 1, history


def aitken_accelerated(x0):
    lam = -1.0 / df(x0)
    x = x0
    history = [x]
    for _ in range(MAX_ITER // 3):
        x1 = x  + lam * f(x)
        x2 = x1 + lam * f(x1)
        denom = x2 - 2 * x1 + x
        if abs(denom) < 1e-14:
            x = x2
            history.append(x)
            break
        x_star = (x2 * x - x1**2) / denom
        history.append(x_star)
        if abs(x_star - x) < TOL and abs(f(x_star)) < TOL:
            break
        dfx = df(x_star)
        if abs(dfx) > 1e-14:
            lam = -1.0 / dfx
        x = x_star
    return x_star, len(history) - 1, history

methods = {
    "Проста ітерація":        simple_iteration,
    "Ньютон":                 newton,
    "Чебишев":                chebyshev,
    "Хорди":                  chord,
    "Парабол":                parabola,
    "Зворотна інтерп.":       reverse_interpolation,
    "Ейткен":                 aitken_accelerated,
}

results = {}
for label, x0 in [("Зростання", x0_rise), ("Спадання", x0_fall)]:
    results[label] = {}
    print(f"\n  ── Корінь ({label}), x₀ = {x0:.3f} ─────────────────────")
    print(f"  {'Метод':<22}  {'x*':>14}  {'|f(x*)|':>12}  {'ітерацій':>10}")
    print(f"  {'-'*62}")
    for name, method in methods.items():
        try:
            xr, nit, hist = method(x0)
            err = abs(f(xr))
            results[label][name] = (xr, nit, hist)
            print(f"  {name:<22}  {xr:14.10f}  {err:12.2e}  {nit:>10}")
        except Exception as e:
            print(f"  {name:<22}  {'ПОМИЛКА':>14}  ({e})")

coeffs = np.array([1.0, -3.0, 4.0, -4.0])   # x³ − 3x² + 4x − 4

with open("lab8/polynomial.txt", "w", encoding="utf-8") as fp:
    fp.write("# Коефіцієнти многочлена: a_n ... a_1 a_0\n")
    fp.write(" ".join(map(str, coeffs)) + "\n")

def read_coeffs(path):
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if not line.startswith("#"):
                return np.array(list(map(float, line.split())))

coeffs = read_coeffs("lab8/polynomial.txt")
n_deg = len(coeffs) - 1

print(f"\n{'='*58}")
print(f"  АЛГЕБРАЇЧНЕ РІВНЯННЯ (ступінь {n_deg})")
print(f"  Коефіцієнти: {coeffs}")
print(f"{'='*58}")

# Схема Горнера
def horner(coeffs, x):
    n = len(coeffs) - 1
    b = np.zeros(n + 1)
    b[0] = coeffs[0]
    for i in range(1, n + 1):
        b[i] = b[i-1] * x + coeffs[i]
    c = np.zeros(n)
    c[0] = b[0]
    for i in range(1, n):
        c[i] = c[i-1] * x + b[i]
    return b[n], c[n-1]

def poly_val(coeffs, x):
    result = complex(coeffs[0])
    for c in coeffs[1:]:
        result = result * x + c
    return result

def newton_horner(coeffs, x0):
    x = x0
    history = [x]
    for _ in range(MAX_ITER):
        px, dpx = horner(coeffs, x)
        if abs(dpx) < 1e-14:
            break
        x_new = x - px / dpx
        history.append(x_new)
        if abs(x_new - x) < TOL and abs(px) < TOL:
            break
        x = x_new
    return x_new, len(history) - 1

# Початкове наближення — шукаємо зміну знаку
xs_alg = np.linspace(-2, 4, 500)
ys_alg = np.array([poly_val(coeffs, xi) for xi in xs_alg])
x0_alg = None
for i in range(len(ys_alg) - 1):
    if ys_alg[i] * ys_alg[i+1] < 0:
        x0_alg = (xs_alg[i] + xs_alg[i+1]) / 2
        break
if x0_alg is None:
    x0_alg = 1.5

real_root, nit_nr = newton_horner(coeffs, x0_alg)

print(f"\n  Метод Ньютона + Горнер:")
print(f"    Початкове наближення: x₀ = {x0_alg:.4f}")
print(f"    Дійсний корінь:  x = {real_root:.10f}")
print(f"    |P(x)| = {abs(poly_val(coeffs, real_root)):.2e}   ітерацій: {nit_nr}")

def lin_method(coeffs, p0=1.0, q0=-1.0):
    a = coeffs.copy()
    n = len(a) - 1
    p, q = p0, q0
    history_pq = [(p, q)]

    for _ in range(MAX_ITER):
        b = np.zeros(n + 1)
        b[0] = a[0]
        b[1] = a[1] - p * b[0]
        for k in range(2, n + 1):
            b[k] = a[k] - p * b[k-1] - q * b[k-2]

        r_val = b[n-1]
        s_val = b[n]

        if abs(r_val) < TOL and abs(s_val) < TOL:
            break

        c = np.zeros(n)
        c[0] = b[0]
        if n >= 2:
            c[1] = b[1] - p * c[0]
        for k in range(2, n):
            c[k] = b[k] - p * c[k-1] - q * c[k-2]

        if n >= 3:
            J = np.array([[-c[n-2], -c[n-3]],
                          [-c[n-1], -c[n-2]]])
        else:
            J = np.array([[-c[0],  0.0],
                          [-c[1], -c[0]]])

        det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        if abs(det) < 1e-14:
            break

        dp = ( J[1,1]*r_val - J[0,1]*s_val) / det
        dq = (-J[1,0]*r_val + J[0,0]*s_val) / det
        p -= dp
        q -= dq
        history_pq.append((p, q))

        if abs(dp) < TOL and abs(dq) < TOL:
            break

    disc = p**2 - 4*q
    if disc >= 0:
        z1 = (-p + np.sqrt(disc)) / 2
        z2 = (-p - np.sqrt(disc)) / 2
    else:
        z1 = complex(-p/2,  np.sqrt(-disc)/2)
        z2 = complex(-p/2, -np.sqrt(-disc)/2)

    return z1, z2, len(history_pq) - 1

z1, z2, nit_lin = lin_method(coeffs, p0=0.0, q0=1.0)

print(f"\n  Метод Ліна (комплексні корені):")
print(f"    z₁ = {z1}")
print(f"    z₂ = {z2}")
print(f"    |P(z₁)| = {abs(poly_val(coeffs, z1.real + 1j*z1.imag if isinstance(z1, complex) else z1)):.2e}")
print(f"    ітерацій: {nit_lin}")

np_roots = np.roots(coeffs)
print(f"\n  Контроль (numpy.roots): {np_roots}")

colors = ['#2563EB','#16A34A','#DC2626','#9333EA','#EA580C','#0891B2','#BE185D']

# Графік 1: трансцендентна функція + корені plot function
x_plot = np.linspace(A - 0.3, B + 0.3, 800)
y_plot = f(x_plot)

plt.figure()
plt.plot(x_plot, y_plot, color='#2563EB', linewidth=2, label='f(x) = x·eˣ − 3·sin(x) − 1')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
for r in roots_approx:
    plt.axvline(r, color='gray', linewidth=0.6, linestyle=':')
for label_k, res in results.items():
    if "Ньютон" in res:
        xr, _, _ = res["Ньютон"]
        plt.scatter([xr], [0], zorder=5, s=60,
                    color='#DC2626' if label_k == "Зростання" else '#16A34A',
                    label=f"Корінь ({label_k}) x={xr:.5f}")
plt.ylim(-8, 8)
plt.xlabel("x");  plt.ylabel("f(x)")
plt.title("Трансцендентна функція f(x) = x·eˣ − 3·sin(x) − 1")
plt.legend(fontsize=9); plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
plt.tight_layout(); plt.savefig("lab8/plot_function.png", dpi=300); plt.close()

# Графік 2: збіжність — кількість ітерацій plot iterations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (label_k, res) in zip(axes, results.items()):
    names = list(res.keys())
    iters = [res[n][1] for n in names]
    clrs  = colors[:len(names)]
    bars  = ax.bar(names, iters, color=clrs, edgecolor='white')
    for bar, it in zip(bars, iters):
        ax.text(bar.get_x() + bar.get_width()/2, it + 0.15, str(it),
                ha='center', va='bottom', fontsize=9)
    ax.set_title(f"Корінь ({label_k})", fontsize=11)
    ax.set_ylabel("Кількість ітерацій")
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, alpha=0.7)
plt.suptitle("Порівняння кількості ітерацій", fontsize=12, y=1.01)
plt.tight_layout(); plt.savefig("lab8/plot_iterations.png", dpi=300, bbox_inches='tight')
plt.close()

# Графік 3: збіжність — похибка по ітераціях (Ньютон vs Ейткен vs Хорди) convergence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
methods_to_plot = ["Проста ітерація", "Ньютон", "Чебишев", "Хорди"]
line_styles = ['-', '--', '-.', ':']

for ax, (label_k, res) in zip(axes, results.items()):
    xr_ref, _, _ = res["Ньютон"]
    for name, ls in zip(methods_to_plot, line_styles):
        if name not in res:
            continue
        _, _, hist = res[name]
        errs = [abs(x - xr_ref) for x in hist if abs(x - xr_ref) > 1e-16]
        if len(errs) < 2:
            continue
        ax.semilogy(range(len(errs)), errs, ls, linewidth=1.8, label=name)
    ax.set_title(f"Корінь ({label_k})", fontsize=11)
    ax.set_xlabel("Ітерація"); ax.set_ylabel("|xₙ − x*|")
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.7)
plt.suptitle("Збіжність методів (лог. шкала)", fontsize=12, y=1.01)
plt.tight_layout(); plt.savefig("lab8/plot_convergence.png", dpi=300, bbox_inches='tight')
plt.close()

# Графік 4: алгебраїчний многочлен polynomial
x_alg = np.linspace(-0.5, 3.5, 500)
y_alg = np.array([poly_val(coeffs, xi).real for xi in x_alg])

plt.figure()
plt.plot(x_alg, y_alg, color='#2563EB', linewidth=2,
         label=r'$P(x)=x^3-3x^2+4x-4$')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.scatter([real_root], [0], color='#DC2626', zorder=5, s=80,
            label=f'Дійсний корінь x={real_root:.5f}')
plt.xlabel("x"); plt.ylabel("P(x)")
plt.title("Алгебраїчне рівняння $P(x) = x^3 - 3x^2 + 4x - 4$")
plt.legend(); plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)
plt.tight_layout(); plt.savefig("lab8/plot_polynomial.png", dpi=300); plt.close()

# Графік 5: комплексні корені на комплексній площині complex roots
all_roots_alg = [real_root, z1, z2]
plt.figure(figsize=(6, 6))
for i, r in enumerate(all_roots_alg):
    re = r.real if isinstance(r, complex) else float(r)
    im = r.imag if isinstance(r, complex) else 0.0
    color = '#DC2626' if im == 0 else '#2563EB'
    marker = 'D' if im == 0 else 'o'
    lbl = f'x={re:.4f}' if im == 0 else f'z={re:.4f}{"+" if im>=0 else ""}{im:.4f}i'
    plt.scatter([re], [im], color=color, s=100, marker=marker, zorder=5, label=lbl)
plt.axhline(0, color='black', linewidth=0.6); plt.axvline(0, color='black', linewidth=0.6)
plt.xlabel("Re"); plt.ylabel("Im")
plt.title("Корені $P(x) = x^3 - 3x^2 + 4x - 4$ на комплексній площині")
plt.legend(); plt.grid(True, linestyle='--', linewidth=0.4, alpha=0.7)
plt.tight_layout(); plt.savefig("lab8/plot_complex_roots.png", dpi=300); plt.close()