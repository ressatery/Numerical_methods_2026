import numpy as np

n = 100
np.random.seed(42)
eps_target = 1e-14

A = np.random.rand(n, n) * 10 - 5
X_true = np.full(n, 2.5)
B = A @ X_true

np.savetxt('A.txt', A, fmt='%.8f')
np.savetxt('B.txt', B, fmt='%.8f')

def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))

        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    return L, U


def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))
    return y


def back_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x


print("\n3. LU-розклад та розв'язок системи ")

L, U = lu_decomposition(A)

y = forward_substitution(L, B)
X = back_substitution(U, y)

residual = np.max(np.abs(A @ X - B))
print(f"\n4. Максимальна нев'язка: {residual:.2e}")

print("\n5. Ітераційне уточнення розв'язку...")

X0 = X.copy()
iter_count = 0
max_iter = 20

for i in range(max_iter):
    R = B - A @ X0
    y = forward_substitution(L, R)
    delta = back_substitution(U, y)
    X0 = X0 + delta
    iter_count += 1

    current_res = np.max(np.abs(A @ X0 - B))
    if current_res < eps_target:
        break

print(f"   Кількість ітерацій: {iter_count}")
print(f"   Фінальна нев'язка: {current_res:.2e}")

np.savetxt('L.txt', L, fmt='%.8f')
np.savetxt('U.txt', U, fmt='%.8f')
np.savetxt('X_solution.txt', X0, fmt='%.10f')
