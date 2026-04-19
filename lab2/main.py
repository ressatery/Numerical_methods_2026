import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['objects']))
            y.append(float(row['fps']))
    return x, y


def divided_differences(x, y):
    n = len(x)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table


def newton_interpolation(x, table, t):
    n = len(x)
    result = table[0][0]
    product = 1.0
    for i in range(1, n):
        product = product * (t - x[i-1])
        result = result + table[0][i] * product
    return result


def finite_differences(y):
    n = len(y)
    delta = [[0.0] * n for _ in range(n)]
    for i in range(n):
        delta[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            delta[i][j] = delta[i+1][j-1] - delta[i][j-1]
    return delta


def factorial_polynomial(x_nodes, y_nodes, t):
    n = len(x_nodes)
    x0, xn = x_nodes[0], x_nodes[-1]
    s = (t - x0) / (xn - x0) * (n - 1)
    delta = finite_differences(y_nodes)
    result = delta[0][0]
    factorial_product = 1.0
    for k in range(1, n):
        factorial_product = factorial_product * (s - (k - 1)) / k
        result = result + factorial_product * delta[0][k]
    return result


def generate_nodes(x_orig, y_orig, n_nodes):
    x0, xn = x_orig[0], x_orig[-1]
    step = (xn - x0) / (n_nodes - 1)
    x_new = [x0 + i * step for i in range(n_nodes)]
    table_orig = divided_differences(x_orig, y_orig)
    y_new = [newton_interpolation(x_orig, table_orig, xi) for xi in x_new]
    return x_new, y_new


csv_data = """objects,fps
500,120
1000,110
1500,90
2000,65
2500,40
"""
with open('data.csv', 'w') as f:
    f.write(csv_data)

x, y = read_data('data.csv')

table = divided_differences(x, y)

print("Таблиця розділених різниць:")
for j in range(len(x)):
    print(f"  Порядок {j}: {table[0][j]:.8f}")

fps_newton    = newton_interpolation(x, table, 1000)
fps_factorial = factorial_polynomial(x, y, 1000)

print(f"\nПрогноз FPS для 1000 об'єктів:")
print(f"  Метод Ньютона:           {fps_newton:.4f}")
print(f"  Факторіальний многочлен: {fps_factorial:.4f}")
print(f"  Різниця між методами:    {abs(fps_newton - fps_factorial):.6f}")

for n_obj in range(100, 2001, 10):
    fps_val = newton_interpolation(x, table, n_obj)
    if fps_val < 60:
        print(f"\nМаксимум {n_obj - 10} об'єктів забезпечують FPS >= 60")
        break

x_interp     = list(range(100, 1601, 10))
y_newton_all = [newton_interpolation(x, table, xi) for xi in x_interp]
y_fact_all   = [factorial_polynomial(x, y, xi) for xi in x_interp]

plt.figure(figsize=(10, 6))
plt.plot(x_interp, y_newton_all, 'b-', linewidth=2, label='Ньютон')
plt.plot(x_interp, y_fact_all, 'g--', linewidth=2, label='Факторіальний')
plt.scatter(x, y, color='red', s=80, zorder=5, label='Дані')
plt.axvline(x=1000, color='orange', linestyle=':', linewidth=1.5, label='n=1000')
plt.axhline(y=60, color='purple', linestyle=':', linewidth=1.5, label='FPS=60')
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.title("Графік 1. FPS від кількості об'єктів")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_fps.png', dpi=100)
plt.close()

colors      = ['blue', 'green', 'orange']
node_counts = [5, 10, 20]

curves = {}
for n_nodes in node_counts:
    xn, yn    = generate_nodes(x, y, n_nodes)
    table_n   = divided_differences(xn, yn)
    curves[n_nodes] = [newton_interpolation(xn, table_n, xi) for xi in x_interp]

plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='red', s=100, zorder=10, label='Дані')
plt.axhline(y=60, color='purple', linestyle=':', linewidth=1.5, label='FPS=60')
for color, n_nodes in zip(colors, node_counts):
    plt.plot(x_interp, curves[n_nodes], color=color, linewidth=1.8, label=f'{n_nodes} вузлів')
plt.xlabel("Кількість об'єктів")
plt.ylabel("FPS")
plt.title("Графік 2. Порівняння при n=5, 10, 20 вузлах")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_comparison.png', dpi=100)
plt.close()

plt.figure(figsize=(12, 6))
for color, n_nodes in zip(colors, node_counts):
    errors = [abs(curves[n_nodes][i] - y_newton_all[i]) for i in range(len(x_interp))]
    plt.plot(x_interp, errors, color=color, linewidth=1.8, label=f'{n_nodes} вузлів')
plt.xlabel("Кількість об'єктів")
plt.ylabel("Похибка")
plt.title("Графік 3. Похибка для n=5, 10, 20 вузлів")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_errors.png', dpi=100)
plt.close()

print("\nЕфект Рунге:")
for n_nodes in node_counts:
    xn, yn  = generate_nodes(x, y, n_nodes)
    table_n = divided_differences(xn, yn)
    val_l   = newton_interpolation(xn, table_n, 150)
    val_r   = newton_interpolation(xn, table_n, 1550)
    ref_l   = newton_interpolation(x, table, 150)
    ref_r   = newton_interpolation(x, table, 1550)
    print(f"  {n_nodes} вузлів: FPS(150)={val_l:.2f} (еталон {ref_l:.2f}), "
          f"FPS(1550)={val_r:.2f} (еталон {ref_r:.2f})")