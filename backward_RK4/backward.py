
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lorenz(u, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = u
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def RK4(u, h):
    k1 = lorenz(u)
    k2 = lorenz(u + 0.5*h*k1)
    k3 = lorenz(u + 0.5*h*k2)
    k4 = lorenz(u + h*k3)

    return u + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def reference_step(u, h):
    sol = solve_ivp(
        lambda t, y: lorenz(y),
        (0, h),
        u,
        method="DOP853",
        rtol=1e-13,
        atol=1e-14
    )
    return sol.y[:, -1]


# Initial condition
u0 = np.array([1.0, 1.0, 1.0])

# Step sizes
h_values = np.logspace(-1.5, -2.5, 12)

defects = []

for h in h_values:
    u_rk4 = RK4(u0, h)
    u_ref = reference_step(u0, h)

    defect = np.linalg.norm(u_rk4 - u_ref)
    defects.append(defect)

defects = np.array(defects)

# Plot
plt.loglog(h_values, defects, "o-", label="One-step defect")
plt.loglog(
    h_values,
    defects[0]*(h_values/h_values[0])**5,
    "--",
    label="O(h^5)"
)

plt.xlabel("Step size h")
plt.ylabel("One-step defect")
plt.title("Lorenz RK4 Local Truncation Error")
plt.legend()
plt.grid(True, which="both")
plt.show()
