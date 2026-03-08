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
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return x, y


def form_matrix(x, m):
    n = m + 1
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(x[k] ** (i + j) for k in range(len(x)))
    return A

def form_vector(x, y, m):
    n = m + 1
    b = [0.0] * n
    for i in range(n):
        b[i] = sum(y[k] * x[k] ** i for k in range(len(x)))
    return b


def gauss_solve(A, b):
    n = len(b)
    A = [row[:] for row in A]
    b = b[:]

    for k in range(n):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]):
                max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]

        for i in range(k + 1, n):
            if A[k][k] == 0:
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]

    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        x_sol[i] = b[i]
        for j in range(i + 1, n):
            x_sol[i] -= A[i][j] * x_sol[j]
        x_sol[i] /= A[i][i]

    return x_sol


def polynomial(x_val, coef):
    result = 0.0
    for i, c in enumerate(coef):
        result += c * (x_val ** i)
    return result

def variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n


csv_data = """Month,Temp
1,-2
2,0
3,5
4,10
5,15
6,20
7,23
8,22
9,17
10,10
11,5
12,0
13,-10
14,3
15,7
16,13
17,19
18,20
19,22
20,21
21,18
22,15
23,10
24,3
"""
with open('data.csv', 'w') as f:
    f.write(csv_data)

x, y = read_data('data.csv')

variances = []
degrees = list(range(1, 8))

for m in degrees:
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(A, b_vec)
    y_approx = [polynomial(xi, coef) for xi in x]
    var = variance(y, y_approx)
    variances.append(var)
    print(f"Степінь {m}: дисперсія = {var:.4f}")

drops = [variances[i - 1] - variances[i] for i in range(1, len(variances))]
optimal_m = degrees[drops.index(max(drops)) + 1]
print(f"\nОптимальний степінь: {optimal_m}")

A = form_matrix(x, optimal_m)
b_vec = form_vector(x, y, optimal_m)
coef = gauss_solve(A, b_vec)
y_approx = [polynomial(xi, coef) for xi in x]

x_future = [25, 26, 27]
y_future = [polynomial(xi, coef) for xi in x_future]
print(f"\nПрогноз на наступні 3 місяці:")
for xi, yi in zip(x_future, y_future):
    print(f"  Місяць {xi}: {yi:.2f}°C  (екстраполяція, може бути неточною)")

errors = [y[i] - y_approx[i] for i in range(len(y))]

plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='red', s=60, zorder=5, label='Фактичні дані')
plt.plot(x, y_approx, 'b-', linewidth=2, label=f'Апроксимація (степінь {optimal_m})')
plt.scatter(x_future, y_future, color='orange', s=80, zorder=5, marker='^', label='Прогноз')
plt.xlabel('Місяць')
plt.ylabel('Температура (°C)')
plt.title('Графік 1. Апроксимація температури методом найменших квадратів')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_approx.png', dpi=100)
plt.close()

plt.figure(figsize=(10, 4))
plt.bar(degrees, variances, color='steelblue', edgecolor='black')
plt.xlabel('Степінь полінома')
plt.ylabel('Дисперсія')
plt.title('Графік 2. Дисперсія від степені полінома')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('plot_variance.png', dpi=100)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(x, errors, 'r-o', linewidth=1.5, markersize=4)
plt.axhline(y=0, color='black', linewidth=1)
plt.xlabel('Місяць')
plt.ylabel('Похибка')
plt.title(f'Графік 3. Похибка апроксимації (степінь {optimal_m})')
plt.grid(True)
plt.tight_layout()
plt.savefig('plot_error.png', dpi=100)
plt.close()