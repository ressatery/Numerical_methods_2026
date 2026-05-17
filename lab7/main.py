import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)          
n = 100
eps_target = 1e-14

A = np.random.uniform(-1, 1, (n, n))
for i in range(n):
    off_diag_sum = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    A[i, i] = off_diag_sum + np.random.uniform(1.0, 2.0)

x_true = np.full(n, 2.5)
B = A @ x_true

np.savetxt('A.txt', A, fmt='%.15f')
np.savetxt('B.txt', B, fmt='%.15f')

def read_matrix(filename):
    return np.loadtxt(filename)

def read_vector(filename):
    return np.loadtxt(filename)

def matrix_vector_product(A, x):
    return A @ x

def vector_norm(x):
    return np.max(np.abs(x))

def matrix_norm(A):
    return np.max(np.sum(np.abs(A), axis=1))

def simple_iteration_method(A, b, x0, tau, eps=1e-14, max_iter=20000, return_history=False):
    n = len(b)
    C = np.eye(n) - tau * A
    d = tau * b
    x = x0.copy()
    history = []
    for k in range(max_iter):
        x_new = C @ x + d
        delta = vector_norm(x_new - x)
        if return_history:
            history.append(delta)
        if delta < eps:
            residual = vector_norm(matrix_vector_product(A, x_new) - b)
            return (x_new, k + 1, residual, history) if return_history else (x_new, k + 1, residual)
        x = x_new
    residual = vector_norm(matrix_vector_product(A, x) - b)
    return (x, max_iter, residual, history) if return_history else (x, max_iter, residual)

def jacobi_method(A, b, x0, eps=1e-14, max_iter=20000, return_history=False):
    n = len(b)
    D_inv = np.diag(1.0 / np.diag(A))
    R = A - np.diag(np.diag(A))
    x = x0.copy()
    history = []
    for k in range(max_iter):
        x_new = D_inv @ (b - R @ x)
        delta = vector_norm(x_new - x)
        if return_history:
            history.append(delta)
        if delta < eps:
            residual = vector_norm(matrix_vector_product(A, x_new) - b)
            return (x_new, k + 1, residual, history) if return_history else (x_new, k + 1, residual)
        x = x_new
    residual = vector_norm(matrix_vector_product(A, x) - b)
    return (x, max_iter, residual, history) if return_history else (x, max_iter, residual)

def seidel_method(A, b, x0, eps=1e-14, max_iter=20000, return_history=False):
    n = len(b)
    x = x0.copy()
    history = []
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s = b[i]
            for j in range(i):
                s -= A[i, j] * x_new[j]
            for j in range(i + 1, n):
                s -= A[i, j] * x[j]
            x_new[i] = s / A[i, i]
        delta = vector_norm(x_new - x)
        if return_history:
            history.append(delta)
        if delta < eps:
            residual = vector_norm(matrix_vector_product(A, x_new) - b)
            return (x_new, k + 1, residual, history) if return_history else (x_new, k + 1, residual)
        x = x_new
    residual = vector_norm(matrix_vector_product(A, x) - b)
    return (x, max_iter, residual, history) if return_history else (x, max_iter, residual)

A = read_matrix('A.txt')
B = read_vector('B.txt')
x0 = np.ones(n)

max_row_sum = matrix_norm(A)
tau = 0.99 / max_row_sum
print(f"\nОбрано τ = {tau:.6f}")
C = np.eye(n) - tau * A
norm_C = matrix_norm(C)
print(f"Норма матриці C (||C||_∞) = {norm_C:.6f} < 1 → збіжність гарантована")

x_simple, it_simple, res_simple, hist_simple = simple_iteration_method(A, B, x0, tau, return_history=True)
x_jacobi, it_jacobi, res_jacobi, hist_jacobi = jacobi_method(A, B, x0, return_history=True)
x_seidel, it_seidel, res_seidel, hist_seidel = seidel_method(A, B, x0, return_history=True)

print("\n=== РЕЗУЛЬТАТИ ===")
print(f"Метод простої ітерації : {it_simple:3d} ітерацій, нев’язка = {res_simple:.2e}")
print(f"Метод Якобі            : {it_jacobi:3d} ітерацій, нев’язка = {res_jacobi:.2e}")
print(f"Метод Зейделя          : {it_seidel:3d} ітерацій, нев’язка = {res_seidel:.2e}")

print("\nПерші 5 компонент розв’язку (метод Зейделя):")
print(np.round(x_seidel[:5], decimals=10))

plt.figure(figsize=(10, 6))
plt.semilogy(range(1, len(hist_simple)+1), hist_simple, 'b-', linewidth=2, label='Проста ітерація')
plt.semilogy(range(1, len(hist_jacobi)+1), hist_jacobi, 'g-', linewidth=2, label='Якобі')
plt.semilogy(range(1, len(hist_seidel)+1), hist_seidel, 'r-', linewidth=2, label='Зейдель')
plt.axhline(y=1e-14, color='k', linestyle='--', label='Задана точність ε = 10⁻¹⁴')
plt.xlabel('Номер ітерації')
plt.ylabel('||x^(k+1) − x^k||_∞  (логарифмічна шкала)')
plt.title('Порівняння швидкості збіжності ітераційних методів')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
print("\nГрафік convergence_plot.png збережено")

plt.figure(figsize=(8, 5))
methods = ['Проста\nітерація', 'Якобі', 'Зейдель']
iters = [it_simple, it_jacobi, it_seidel]
bars = plt.bar(methods, iters, color=['blue', 'green', 'red'])
plt.ylabel('Кількість ітерацій')
plt.title('Порівняння кількості ітерацій для досягнення точності 10⁻¹⁴')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars, iters):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontsize=13, fontweight='bold')
plt.savefig('iterations_bar.png', dpi=300, bbox_inches='tight')
print("Графік iterations_bar.png збережено")
