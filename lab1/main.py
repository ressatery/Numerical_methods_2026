import requests
import numpy as np
import matplotlib.pyplot as plt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Радіус Землі в метрах
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

url = ("https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106")

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

print("Кількість вузлів:", n)
print("\nТабуляція вузлів:")
print("№ | Latitude   | Longitude  | Elevation (m)")
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

with open("gps_data.txt", "w") as f:
    f.write("№ | Latitude   | Longitude  | Elevation (m)\n")
    for i, point in enumerate(results):
        f.write(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}\n")

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0]
for i in range(1, n):
    d = haversine(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print("№ | Distance (m) | Elevation (m)")
for i in range(n):
    print(f"{i:2d} | {distances[i]:10.2f} | {elevations[i]:8.2f}")

def cubic_spline(x, y):
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]

    # коефіцієнти тридіагональної матриці
    alpha = [0.0] * (n + 1)
    beta  = [0.0] * (n + 1)
    gamma = [0.0] * (n + 1)
    delta = [0.0] * (n + 1)

    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i]  = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # пряма
    A = [0.0] * n
    B = [0.0] * n

    A[0] = -gamma[1] / beta[1]
    B[0] =  delta[1] / beta[1]
    for i in range(1, n - 1):
        denom = alpha[i + 1] * A[i - 1] + beta[i + 1]
        A[i] = -gamma[i + 1] / denom
        B[i] = (delta[i + 1] - alpha[i + 1] * B[i - 1]) / denom

    # зворотна
    c = [0.0] * (n + 1)
    c[n] = 0.0
    for i in range(n - 1, 0, -1):
        c[i] = A[i - 1] * c[i + 1] + B[i - 1]
    c[0] = 0.0

    # обчислення b та d через знайдені c
    a_coef = list(y[:-1]) if not isinstance(y, list) else y[:-1]
    b_coef = [0.0] * n
    d_coef = [0.0] * n
    for i in range(n):
        b_coef[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d_coef[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a_coef, b_coef, c[:-1], d_coef

xx = distances
yy = elevations
a, b, c, d = cubic_spline(xx, yy)

print("\nКоефіцієнти сплайнів (a, b, c, d для кожного інтервалу):")
for i in range(len(a)):
    print(f"Інтервал {i+1:2d}: a={a[i]:.4f}, b={b[i]:.6f}, c={c[i]:.6f}, d={d[i]:.8f}")

def evaluate_spline(x_val, xx, a, b, c, d):
    for i in range(len(xx) - 1):
        if xx[i] <= x_val <= xx[i + 1]:
            dx = x_val - xx[i]
            return a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3
    return yy[-1] if x_val > xx[-1] else yy[0]

xx_full = np.linspace(min(xx), max(xx), 1000)
yy_full = np.array([evaluate_spline(xv, xx, a, b, c, d) for xv in xx_full])

total_length  = distances[-1]
total_ascent  = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print(f"\nЗагальна довжина маршруту (м): {total_length:.2f}")
print(f"Сумарний набір висоти (м):      {total_ascent:.2f}")
print(f"Сумарний спуск (м):             {total_descent:.2f}")

grad_full = np.gradient(yy_full, xx_full) * 100
print(f"\nМаксимальний підйом (%):        {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%):         {np.min(grad_full):.2f}")
print(f"Середній градієнт (%):          {np.mean(np.abs(grad_full)):.2f}")
steep = np.sum(np.abs(grad_full) > 15) / len(grad_full) * 100
print(f"Ділянки з крутизною >15% (%):   {steep:.2f}")

mass   = 80
g      = 9.81
energy = mass * g * total_ascent
print(f"\nМеханічна робота (Дж):  {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал):         {energy / 4184:.2f}")

def subsample_spline(k, distances, elevations):
    idx = np.round(np.linspace(0, len(distances) - 1, k)).astype(int)
    x_sub = [distances[i] for i in idx]
    y_sub = [elevations[i] for i in idx]
    a_s, b_s, c_s, d_s = cubic_spline(x_sub, y_sub)
    yy_approx = np.array([evaluate_spline(xv, x_sub, a_s, b_s, c_s, d_s) for xv in xx_full])
    return yy_approx

yy_10 = subsample_spline(10, distances, elevations)
yy_15 = subsample_spline(15, distances, elevations)
yy_20 = subsample_spline(20, distances, elevations)

err_10 = np.abs(yy_full - yy_10)
err_15 = np.abs(yy_full - yy_15)
err_20 = np.abs(yy_full - yy_20)

print("\n*** 10 вузлів")
print(f"Максимальна похибка: {np.max(err_10):.6f}")
print(f"Середня похибка:     {np.mean(err_10):.6f}")

print("\n** 15 вузлів")
print(f"Максимальна похибка: {np.max(err_15):.6f}")
print(f"Середня похибка:     {np.mean(err_15):.6f}")

print("\n* 20 вузлів")
print(f"Максимальна похибка: {np.max(err_20):.6f}")
print(f"Середня похибка:     {np.mean(err_20):.6f}")

plt.plot(xx, yy, 'o', label="Bузли (21)")
plt.plot(xx_full, yy_full, '-', label="Кубічний сплайн (еталон)")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Профіль висоти маршруту")
plt.legend()
plt.show()

plt.plot(xx_full, yy_full,  label="21 вузол (еталон)", linewidth=2)
plt.plot(xx_full, yy_10,    label="10 вузлів",  linestyle="--")
plt.plot(xx_full, yy_15,    label="15 вузлів",  linestyle="--")
plt.plot(xx_full, yy_20,    label="20 вузлів",  linestyle="--")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.title("Вплив кількості вузлів")
plt.legend()
plt.show()

plt.plot(xx_full, err_10, label="10 вузлів")
plt.plot(xx_full, err_15, label="15 вузлів")
plt.plot(xx_full, err_20, label="20 вузлів")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Абсолютна похибка (м)")
plt.title("Похибка апроксимації")
plt.legend()
plt.show()