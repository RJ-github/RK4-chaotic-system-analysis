import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp


# ============================================================
# NumPy float32 / float64 Lorenz
# ============================================================

def lorenz_np(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return np.array([dx, dy, dz], dtype=state.dtype)


def rk4_step_np(state, dt):
    dt = state.dtype.type(dt)

    k1 = lorenz_np(state)
    k2 = lorenz_np(state + 0.5 * dt * k1)
    k3 = lorenz_np(state + 0.5 * dt * k2)
    k4 = lorenz_np(state + dt * k3)

    return state + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


def simulate_np(initial_state, dtype, dt, T_final):
    steps = int(T_final / dt)

    state = np.array(initial_state, dtype=dtype)

    trajectory = np.zeros(
        (steps, 3),
        dtype=dtype
    )

    for i in range(steps):
        trajectory[i] = state
        state = rk4_step_np(state, dt)

    return trajectory


# ============================================================
# mpmath high-precision Lorenz
# ============================================================

def lorenz_mp(state):
    sigma = mp.mpf("10.0")
    rho = mp.mpf("28.0")
    beta = mp.mpf("8.0") / mp.mpf("3.0")

    x, y, z = state

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return [dx, dy, dz]


def rk4_step_mp(state, dt):
    k1 = lorenz_mp(state)

    k2_state = [
        state[i]
        + mp.mpf("0.5") * dt * k1[i]
        for i in range(3)
    ]

    k2 = lorenz_mp(k2_state)

    k3_state = [
        state[i]
        + mp.mpf("0.5") * dt * k2[i]
        for i in range(3)
    ]

    k3 = lorenz_mp(k3_state)

    k4_state = [
        state[i] + dt * k3[i]
        for i in range(3)
    ]

    k4 = lorenz_mp(k4_state)

    return [
        state[i]
        + (dt / mp.mpf("6.0"))
        * (
            k1[i]
            + 2 * k2[i]
            + 2 * k3[i]
            + k4[i]
        )
        for i in range(3)
    ]


def simulate_mp(initial_state, dt, T_final, dps=500):
    mp.mp.dps = dps

    steps = int(T_final / dt)

    dt_mp = mp.mpf(str(dt))

    state = [
        mp.mpf(str(v))
        for v in initial_state
    ]

    # Stored as float64 for easy NumPy comparison.
    # Integration itself still occurs at arbitrary precision.
    trajectory = np.zeros(
        (steps, 3),
        dtype=np.float64
    )

    for i in range(steps):
        trajectory[i] = [
            float(state[0]),
            float(state[1]),
            float(state[2])
        ]

        state = rk4_step_mp(
            state,
            dt_mp
        )

    return trajectory


# ============================================================
# Prediction horizon analysis
# ============================================================

def prediction_horizon(error, time, epsilon):
    """
    Finds the first time that the error exceeds epsilon.
    """

    indices = np.where(
        error > epsilon
    )[0]

    if len(indices) == 0:
        return None

    return time[indices[0]]


def compute_horizons_for_thresholds(
    error,
    time,
    epsilons
):
    horizons = {}

    for epsilon in epsilons:
        horizons[epsilon] = prediction_horizon(
            error,
            time,
            epsilon
        )

    return horizons


# ============================================================
# Error-growth analysis
# ============================================================

def analyze_error_growth(
    time,
    error,
    min_error=1e-6,
    max_error=1.0
):
    """
    Fits the approximately exponential region of the error:

        E(t) ~= E0 * exp(lambda * t)

    Taking the natural logarithm gives:

        ln(E) ~= lambda*t + ln(E0)

    Therefore the slope of ln(error) versus time
    estimates the exponential error-growth rate.
    """

    mask = (
        np.isfinite(error)
        & (error > min_error)
        & (error < max_error)
    )

    t_fit = time[mask]
    error_fit = error[mask]

    if len(t_fit) < 2:
        return None

    log_error = np.log(error_fit)

    slope, intercept = np.polyfit(
        t_fit,
        log_error,
        1
    )

    predicted_log_error = (
        slope * t_fit
        + intercept
    )

    predicted_error = np.exp(
        predicted_log_error
    )

    # R^2 calculation
    ss_res = np.sum(
        (
            log_error
            - predicted_log_error
        ) ** 2
    )

    ss_tot = np.sum(
        (
            log_error
            - np.mean(log_error)
        ) ** 2
    )

    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "time_fit": t_fit,
        "error_fit": error_fit,
        "predicted_error": predicted_error
    }


def instantaneous_growth_rate(
    time,
    error
):
    """
    Estimates:

        d(ln(E)) / dt

    which represents the local rate at which
    numerical error is growing.
    """

    safe_error = np.maximum(
        error,
        np.finfo(np.float64).tiny
    )

    log_error = np.log(
        safe_error
    )

    growth_rate = np.gradient(
        log_error,
        time
    )

    return growth_rate


# ============================================================
# Main experiment
# ============================================================

def run_experiment(
    initial_state,
    dt=0.01,
    T_final=100.0,
    dps=500,
    epsilons=None
):
    if epsilons is None:
        epsilons = [
            0.1,
            0.5,
            1.0,
            2.0,
            5.0
        ]

    print(
        "Computing high-precision reference..."
    )

    traj_mp = simulate_mp(
        initial_state,
        dt,
        T_final,
        dps
    )

    print(
        "Computing float32 trajectory..."
    )

    traj32 = simulate_np(
        initial_state,
        np.float32,
        dt,
        T_final
    )

    print(
        "Computing float64 trajectory..."
    )

    traj64 = simulate_np(
        initial_state,
        np.float64,
        dt,
        T_final
    )

    time = (
        np.arange(len(traj_mp))
        * dt
    )

    # Euclidean error
    error32 = np.linalg.norm(
        traj32.astype(np.float64)
        - traj_mp,
        axis=1
    )

    error64 = np.linalg.norm(
        traj64
        - traj_mp,
        axis=1
    )

    horizons32 = (
        compute_horizons_for_thresholds(
            error32,
            time,
            epsilons
        )
    )

    horizons64 = (
        compute_horizons_for_thresholds(
            error64,
            time,
            epsilons
        )
    )

    # Exponential-growth analysis
    analysis32 = analyze_error_growth(
        time,
        error32
    )

    analysis64 = analyze_error_growth(
        time,
        error64
    )

    return {
        "time": time,
        "traj_mp": traj_mp,
        "traj32": traj32,
        "traj64": traj64,
        "error32": error32,
        "error64": error64,
        "horizons32": horizons32,
        "horizons64": horizons64,
        "analysis32": analysis32,
        "analysis64": analysis64,
        "epsilons": epsilons
    }


# ============================================================
# Print results
# ============================================================

def print_results(results):
    print()

    print(
        "epsilon\t\t"
        "float32 horizon\t\t"
        "float64 horizon"
    )

    print("-" * 60)

    for epsilon in results["epsilons"]:
        h32 = (
            results["horizons32"][epsilon]
        )

        h64 = (
            results["horizons64"][epsilon]
        )

        h32_text = (
            "never"
            if h32 is None
            else f"{h32:.5f}"
        )

        h64_text = (
            "never"
            if h64 is None
            else f"{h64:.5f}"
        )

        print(
            f"{epsilon:<10}\t"
            f"{h32_text:<16}\t"
            f"{h64_text:<16}"
        )


def print_growth_analysis(results):
    print()
    print("Error Growth Analysis")
    print("-" * 50)

    analysis32 = results[
        "analysis32"
    ]

    analysis64 = results[
        "analysis64"
    ]

    if analysis32 is not None:
        print(
            "float32 growth rate: "
            f"{analysis32['slope']:.6f}"
        )

        print(
            "float32 R^2: "
            f"{analysis32['r_squared']:.6f}"
        )

    else:
        print(
            "Could not fit float32 "
            "exponential growth region."
        )

    print()

    if analysis64 is not None:
        print(
            "float64 growth rate: "
            f"{analysis64['slope']:.6f}"
        )

        print(
            "float64 R^2: "
            f"{analysis64['r_squared']:.6f}"
        )

    else:
        print(
            "Could not fit float64 "
            "exponential growth region."
        )


# ============================================================
# Plot Lorenz attractor
# ============================================================

def plot_lorenz_attractor(
    trajectory
):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(
        projection="3d"
    )

    ax.plot(
        x,
        y,
        z
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(
        "Lorenz Attractor"
    )

    plt.show()


# ============================================================
# Plot raw error growth
# ============================================================

def plot_error_growth(results):
    time = results["time"]

    plt.figure()

    plt.plot(
        time,
        results["error32"],
        label=(
            "float32 vs "
            "high precision"
        )
    )

    plt.plot(
        time,
        results["error64"],
        label=(
            "float64 vs "
            "high precision"
        )
    )

    for epsilon in results[
        "epsilons"
    ]:
        plt.axhline(
            epsilon,
            linestyle="--",
            linewidth=0.8
        )

    plt.yscale("log")

    plt.xlabel("Time")
    plt.ylabel("Error")

    plt.title(
        "Error Growth with "
        "Multiple Prediction Thresholds"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


# ============================================================
# Prediction horizon versus threshold
# ============================================================

def plot_horizon_vs_threshold(
    results
):
    epsilons = results[
        "epsilons"
    ]

    h32 = [
        results["horizons32"][eps]
        for eps in epsilons
    ]

    h64 = [
        results["horizons64"][eps]
        for eps in epsilons
    ]

    plt.figure()

    plt.plot(
        epsilons,
        h32,
        marker="o",
        label="float32"
    )

    plt.plot(
        epsilons,
        h64,
        marker="o",
        label="float64"
    )

    plt.xscale("log")

    plt.xlabel(
        "Error threshold epsilon"
    )

    plt.ylabel(
        "Prediction horizon"
    )

    plt.title(
        "Prediction Horizon "
        "vs Error Threshold"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


# ============================================================
# Exponential error-growth fitting plot
# ============================================================

def plot_error_growth_analysis(
    results
):
    time = results["time"]

    error32 = results["error32"]
    error64 = results["error64"]

    analysis32 = results[
        "analysis32"
    ]

    analysis64 = results[
        "analysis64"
    ]

    plt.figure()

    plt.semilogy(
        time,
        error32,
        label="float32 error"
    )

    plt.semilogy(
        time,
        error64,
        label="float64 error"
    )

    if analysis32 is not None:
        plt.semilogy(
            analysis32["time_fit"],
            analysis32[
                "predicted_error"
            ],
            "--",
            label=(
                "float32 fit, "
                f"growth = "
                f"{analysis32['slope']:.3f}"
            )
        )

    if analysis64 is not None:
        plt.semilogy(
            analysis64["time_fit"],
            analysis64[
                "predicted_error"
            ],
            "--",
            label=(
                "float64 fit, "
                f"growth = "
                f"{analysis64['slope']:.3f}"
            )
        )

    plt.xlabel("Time")
    plt.ylabel("Error")

    plt.title(
        "Exponential Growth "
        "of Numerical Error"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


# ============================================================
# Local / instantaneous error growth rate
# ============================================================

def plot_growth_rate(results):
    time = results["time"]

    growth32 = (
        instantaneous_growth_rate(
            time,
            results["error32"]
        )
    )

    growth64 = (
        instantaneous_growth_rate(
            time,
            results["error64"]
        )
    )

    plt.figure()

    plt.plot(
        time,
        growth32,
        label="float32"
    )

    plt.plot(
        time,
        growth64,
        label="float64"
    )

    plt.xlabel("Time")

    plt.ylabel(
        r"$d(\ln E)/dt$"
    )

    plt.title(
        "Instantaneous Numerical "
        "Error Growth Rate"
    )

    plt.legend()
    plt.grid(True)

    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    initial_state = [
        1.0,
        1.0,
        1.0
    ]

    dt = 0.01
    T_final = 100.0
    dps = 80

    epsilons = [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0
    ]

    results = run_experiment(
        initial_state,
        dt,
        T_final,
        dps,
        epsilons
    )

    # Print numerical results
    print_results(results)

    print_growth_analysis(
        results
    )

    # Lorenz attractor
    plot_lorenz_attractor(
        results["traj64"]
    )

    # Error versus time
    plot_error_growth(
        results
    )

    # Prediction horizon
    plot_horizon_vs_threshold(
        results
    )

    # Exponential error fit
    plot_error_growth_analysis(
        results
    )

    # Local error-growth rate
    plot_growth_rate(
        results
    )


if __name__ == "__main__":
    main()
