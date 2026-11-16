
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ==================================================
# 1. LORENZ SYSTEM
# ==================================================

sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0


def lorenz(t, u, rho):
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


def df_drho(u):
    return np.array([0.0, u[0], 0.0])


# ==================================================
# 2. REFERENCE TRAJECTORY
# ==================================================

def reference_trajectory(rho, T, dt):
    """
    Generate states at midpoint times:

        t_j = (j + 1/2) * dt

    First discard a transient.
    """

    m = round(T / dt)
    T = m * dt

    u0 = np.array([1.0, 1.0, 1.0])

    # Remove transient
    transient = solve_ivp(
        lorenz, (0, 50), u0,
        args=(rho,),
        rtol=1e-10,
        atol=1e-12
    )

    u_start = transient.y[:, -1]

    times = (np.arange(m) + 0.5) * dt

    sol = solve_ivp(
        lorenz, (0, T), u_start,
        args=(rho,),
        t_eval=times,
        rtol=1e-10,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.y.T


# ==================================================
# 3. BUILD LSS CONSTRAINT MATRIX
# ==================================================

def build_lss_matrix(U, rho, dt, alpha):
    """
    Equation (17) and Equation (20).

    Unknown vector:

        x = [v_0, ..., v_(m-1),
             alpha*eta_1, ..., alpha*eta_(m-1)]

    Constraint:

        B @ x = b

    Each v_j has 3 components.
    """

    m, n = U.shape

    rows = n * (m - 1)
    cols = n * m + (m - 1)

    B = lil_matrix((rows, cols))
    b = np.zeros(rows)

    I = np.eye(n)

    for i in range(1, m):

        u_left = U[i - 1]
        u_right = U[i]

        A_left = jacobian(u_left, rho)
        A_right = jacobian(u_right, rho)

        # Equation (17)
        E = -I / dt - A_left
        G = I / dt - A_right

        f_i = (u_right - u_left) / dt

        b_i = 0.5 * (
            df_drho(u_left) +
            df_drho(u_right)
        )

        # Rows belonging to constraint i
        r = slice(n*(i-1), n*i)

        # Left sensitivity v_(i-1)
        c_left = slice(n*(i-1), n*i)

        # Right sensitivity v_i
        c_right = slice(n*i, n*(i+1))

        # Time dilation variable
        c_eta = n*m + (i-1)

        B[r, c_left] = E
        B[r, c_right] = G
        B[r, c_eta] = f_i / alpha

        b[r] = b_i

    return B.tocsr(), b


# ==================================================
# 4. SOLVE LSS PROBLEM
# ==================================================

def solve_lss(U, rho, dt, alpha=10.0):

    m, n = U.shape

    B, b = build_lss_matrix(
        U, rho, dt, alpha
    )

    # Equation (19)
    M = (B @ B.T).tocsc()

    w = spsolve(M, b)

    # Minimum-norm solution
    x = np.asarray(B.T @ w).ravel()

    # Recover v and eta
    V = x[:n*m].reshape(m, n)

    eta = x[n*m:] / alpha

    # Verify constraint residual
    residual = np.linalg.norm(B @ x - b)
    relative_residual = residual / max(
        np.linalg.norm(b), 1e-15
    )

    print("Relative constraint residual:",
          relative_residual)

    return V, eta


# ==================================================
# 5. COMPUTE SENSITIVITY
# ==================================================

def lss_sensitivity(U, V, eta):
    """
    Equation (22), specialized to J(u) = z.
    """

    z = U[:, 2]
    vz = V[:, 2]

    z_mean = np.mean(z)

    # Midpoint contribution
    state_term = np.mean(vz)

    # Time dilation contribution
    z_edges = 0.5 * (z[:-1] + z[1:])

    time_term = np.mean(
        eta * (z_edges - z_mean)
    )

    return state_term + time_term


# ==================================================
# 6. RUN EXPERIMENT
# ==================================================

T = 10.0
dt = 0.02
alpha = 10.0

U = reference_trajectory(rho, T, dt)

V, eta = solve_lss(
    U, rho, dt, alpha
)

sensitivity = lss_sensitivity(U, V, eta)

print("Mean z:", np.mean(U[:, 2]))
print("LSS sensitivity:", sensitivity)


# ==================================================
# 7. PLOTS
# ==================================================

times = (np.arange(len(U)) + 0.5) * dt

fig, axes = plt.subplots(3, 1, figsize=(10, 9))

axes[0].plot(times, U[:, 2])
axes[0].set_ylabel("z(t)")
axes[0].set_title("Reference Lorenz Trajectory")

axes[1].plot(times, V[:, 2])
axes[1].set_ylabel("v_z(t)")
axes[1].set_title("LSS Shadowing Sensitivity")

axes[2].plot(times[1:], eta)
axes[2].set_ylabel("eta(t)")
axes[2].set_xlabel("Time")
axes[2].set_title("Time Dilation Sensitivity")

for ax in axes:
    ax.grid(True)

plt.tight_layout()
plt.show()


T_values = [5, 10, 20, 40, 80]
results = []

for T in T_values:

    U = reference_trajectory(rho, T, dt)
    V, eta = solve_lss(U, rho, dt, alpha)

    dJ = lss_sensitivity(U, V, eta)
    results.append(dJ)

    print(f"T = {T:5.1f}, sensitivity = {dJ:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(T_values, results, "o-")
plt.axhline(1.0, color="gray", linestyle="--",
            label="Reference near 1")
plt.xlabel("Trajectory length T")
plt.ylabel("d<z>/d(rho)")
plt.title("LSS Sensitivity vs Trajectory Length")
plt.legend()
plt.grid(True)
plt.show()
