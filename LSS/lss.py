
import numpy as np
import matplotlib.pyplot as plt

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0


def lorenz(u, rho):
    x, y, z = u
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def jacobian(u, rho):
    x, y, z = u
    return np.array([
        [-sigma, sigma, 0],
        [rho - z, -1, -x],
        [y, x, -beta]
    ])


def parameter_derivative(u):
    x, y, z = u
    return np.array([0, x, 0])


def combined_system(Y, rho):
    u = Y[:3]
    v = Y[3:]

    du = lorenz(u, rho)
    dv = jacobian(u, rho) @ v + parameter_derivative(u)

    return np.concatenate([du, dv])


def rk4_step(f, y, dt, *args):
    k1 = f(y, *args)
    k2 = f(y + dt*k1/2, *args)
    k3 = f(y + dt*k2/2, *args)
    k4 = f(y + dt*k3, *args)

    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def integrate(f, y0, T, dt, *args):
    n = round(T / dt)
    t = np.linspace(0, n*dt, n+1)

    Y = np.zeros((n+1, len(y0)))
    Y[0] = y0

    for i in range(n):
        Y[i+1] = rk4_step(f, Y[i], dt, *args)

    return t, Y


# Initial conditions
u0 = np.array([1.0, 1.0, 1.0])
v0 = np.zeros(3)

Y0 = np.concatenate([u0, v0])

t, Y = integrate(combined_system, Y0, 20.0, 0.01, rho)

u = Y[:, :3]
v = Y[:, 3:]

sensitivity_norm = np.linalg.norm(v, axis=1)

plt.figure(figsize=(9, 5))
plt.semilogy(t, sensitivity_norm + 1e-15)
plt.xlabel("Time")
plt.ylabel("||v(t)||")
plt.title("Growth of Conventional Tangent Sensitivity")
plt.grid(True)
plt.show()


def time_average_z(rho, T, dt, u0):
    t, U = integrate(lorenz, u0, T, dt, rho)
    return np.trapezoid(U[:, 2], t) / T


def finite_difference_sensitivity(rho, h, T, dt, u0):
    zp = time_average_z(rho + h, T, dt, u0)
    zm = time_average_z(rho - h, T, dt, u0)

    return (zp - zm) / (2*h)


def tangent_sensitivity(rho, T, dt, u0):
    Y0 = np.concatenate([u0, np.zeros(3)])

    t, Y = integrate(combined_system, Y0, T, dt, rho)

    vz = Y[:, 5]

    return np.trapezoid(vz, t) / T


T_values = [5, 10, 15, 20, 25, 30]
h = 0.5

fd_results = []
tangent_results = []

for T in T_values:
    fd = finite_difference_sensitivity(rho, h, T, 0.01, u0)
    tangent = tangent_sensitivity(rho, T, 0.01, u0)

    fd_results.append(fd)
    tangent_results.append(tangent)

    print(f"T={T:5.1f} | FD={fd:12.5f} | Tangent={tangent:14.5e}")


plt.figure(figsize=(9, 5))
plt.plot(T_values, fd_results, "o-", label="Finite difference")
plt.plot(T_values, tangent_results, "s-", label="Tangent method")

plt.yscale("symlog", linthresh=1)
plt.xlabel("Averaging time T")
plt.ylabel("Estimated sensitivity")
plt.title("Conventional vs Finite-Difference Sensitivity")
plt.legend()
plt.grid(True)
plt.show()
