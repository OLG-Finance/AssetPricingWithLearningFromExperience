import numpy as np
from typing import Tuple


def post_var(
        sigma_Y_sq: float,
        V_hat: float,
        tau: np.ndarray,
) -> np.ndarray:
    """
    Update V according to Proposition 1

    Args:
        sigma_Y_sq (float): sigma_Y squared
        V_hat (float): initial variance
        tau (np.ndarray): t-s, time since birth

    Returns:
        np.ndarray: shape (Nc)
    """
    V_st = sigma_Y_sq * V_hat / (sigma_Y_sq + V_hat * tau)
    return (
        V_st
    )


def dDelta_st_calculator(
        sigma_Y_sq: float,
        dt: float,
        V_st: np.ndarray,
        Delta_st: np.ndarray,
        dZ_t: float,
) -> np.ndarray:
    """Calculate change in beliefs

    Args:
        sigma_Y_sq (float): sigma_Y squared
        dt (float): dt
        V_st (np.ndarray): posterior variance
        Delta_st (np.ndarray): prior estimation error
        dZ_t (float): shocks to the fundamental

    Returns:
        np.ndarray: shape (Nc)
    """
    dDelta_st = V_st / sigma_Y_sq * (
                - Delta_st * dt + dZ_t
        )
    return dDelta_st.astype(np.float32)


def build_cohorts(
    dZ_build: np.ndarray,
    Nc: int,
    dt: float,
    tau: np.ndarray,
    nu: float,
    beta: float,
    V_hat: float,
    sigma_Y: float,
    N_pre: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy

    Args:
        dZ_build (np.ndarray): shocks to the output for each period, shape (Nc-1, )
        Nc (int): number of cohorts to build up
        dt (float): unit of time
        tau (np.ndarray): time since birth for each cohort
        nu (float):
        beta (float):
        V_hat (float): initial variance of beliefs
        sigma_Y (float): sd of aggregate output growth
        N_pre (int): pre-trading periods

    Returns:
        Delta_s_t (np.ndarray): estimation bias, shape(Nc, )
        eta_st_eta_ss(np.ndarray): shape(Nc, )
        X(np.ndarray):W_s * Xi_s, shape(Nc, )
    """

    # initialize the matrices
    Delta_st = np.zeros(1)
    X = np.ones(1)
    eta_st_eta_ss = np.ones(1)
    sigma_Y_sq = sigma_Y ** 2
    T_hat = N_pre * dt

    for i in range(1, Nc):
        tau_short = tau[-i:]
        # a new cohort born (age 0), get wealth transfer, observe, invest
        dZ_build_t = dZ_build[i - 1]

        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * Delta_st ** 2) * dt
            + Delta_st * dZ_build_t
        )

        # X is the collection of all X_s, s<t.
        X_parts = nu * beta * X * eta_st_eta_ss * np.exp(-nu * beta * tau_short) * dt
        X_t = np.sum(X_parts) / (1 - nu * beta * dt)  # dividing by (1-nu*beta*dt) keeps sum(f_st*dt) at 1

        eta_st_eta_ss = np.append(eta_st_eta_ss, 1)
        X = np.append(X, X_t)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort

        # update beliefs
        V_st = post_var(sigma_Y_sq, V_hat, tau_short)
        dDelta_st = dDelta_st_calculator(sigma_Y_sq, dt, V_st, Delta_st, dZ_build_t)

        if i < N_pre:
            Delta_st += dDelta_st
            Delta_st = np.append(
                Delta_st,
                0
            )  # newborns begin with 0 bias when there are not enough observations
        else:
            init_bias = np.sum(dZ_build[int(i - N_pre):i]) / T_hat
            Delta_st += dDelta_st
            Delta_st = np.append(
                Delta_st,
                init_bias
            )  # newborns begin with N_pre observations of the dividend process

    return (
        Delta_st,
        eta_st_eta_ss,
        X,
    )


def simulate_cohorts(
        bias_vec: np.ndarray,
        dZ: np.ndarray,
        Nt: int,
        Nc: int,
        tau: np.ndarray,
        dt: float,
        nu: float,
        beta: float,
        V_hat: float,
        mu_Y: float,
        sigma_Y: float,
        T_hat: float,
        N_pre: float,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        Delta_st: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """ Simulate the economy forward

    Args:
        bias_vec (np.ndarray): shocks to the output in the build stage for the cohorts born between s=[0, Npre]
        dZ (np.ndarray): shocks to the output for each period, shape (Nt, )
        Nt (int): number of periods in the simulation
        Nc (int): number of cohorts in the economy
        tau (np.ndarray): t-s, shape(Nt)
        dt (float): per unit of time
        beta (np.ndarray): consumption wealth ratio
        nu (float): rate of birth and death
        V_hat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        T_hat (float): pre-trading years
        N_pre (float): pre-trading number of observations
        -*-*-*- from the cohort_builder function -*-*-*-
        eta_st_eta_ss(np.ndarray): shape(Nc, )
        X(np.ndarray): W_s * Xi_s, shape(Nc, )
        Delta_st (np.ndarray): estimation bias, shape(Nc)

    Returns:
        dR (np.ndarray): realized stock returns, shape (Nt, )
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        Delta_bar (np.ndarray): consumption weighted average estimation error of participants, shape(Nt)
        mu_S (np.ndarray): expected equity risk premia, shape(Nt)
        f_mat (np.ndarray): consumption share, shape(Nt, Nc)
        Delta_mat (np.ndarray): standardized estimation error, shape(Nt, Nc)
        pi_mat (np.ndarray): portfolio, shape(Nt, Nc)

    """
    # Initializing variables
    f_mat = np.zeros((Nt, Nc), dtype=np.float16)  # evolution of cohort consumption share
    Delta_mat = np.zeros((Nt, Nc), dtype=np.float16)  # stores bias in beliefs
    pi_mat = np.zeros((Nt, Nc), dtype=np.float16)  # portfolio choice

    # equilibrium terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    mu_S = np.zeros(Nt)
    Delta_bar = np.zeros(Nt)  # consumption weighted estimation error

    sigma_Y_sq = sigma_Y ** 2
    V_st = post_var(sigma_Y_sq, V_hat, tau)

    mu_S_t = 0
    sigma_S_t = np.copy(sigma_Y)

    for i in range(Nt):
        dZ_t = dZ[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = np.copy(
            eta_st_eta_ss * np.exp(
            (-0.5 * Delta_st ** 2) * dt
            + Delta_st * dZ_t
        )
        )

        X_parts = nu * beta * X * eta_st_eta_ss * np.exp(-nu * beta * tau) * dt
        X_t = np.sum(X_parts) / (1 - nu * beta * dt)  # dividing by (1-nu*beta*dt) keeps sum(f_st*dt) at 1

        eta_st_eta_ss = np.append(eta_st_eta_ss[1:], 1.0)
        X = np.copy(
            np.append(X[1:], X_t)
        ) / X_t  # rescale, does not change the relative magnitude of each cohort

        f_st = X_parts / X_t / dt
        f_st = np.append(f_st[1:], nu * beta)

        dR_t = mu_S_t * dt + sigma_S_t * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t

        # update beliefs
        dDelta_st = dDelta_st_calculator(sigma_Y_sq, dt, V_st, Delta_st, dZ_t)

        if i < N_pre - 1:
            init_bias = (np.sum(bias_vec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat
        else:
            init_bias = np.sum(dZ[i + 1 - N_pre: i + 1]) / T_hat

        Delta_st = np.copy(
            np.append(
                Delta_st[1:] + dDelta_st[1:],
                init_bias
            )
        )

        Delta_bar_t = np.average(Delta_st, weights=f_st)
        d_eta_st = np.copy(Delta_st)

        r_t = (
                nu * (1 - beta) + mu_Y + Delta_bar_t
                - sigma_Y_sq
        )
        theta_t = sigma_Y - Delta_bar_t
        mu_S_t = sigma_S_t * theta_t + r_t
        pi_st = (d_eta_st + theta_t) / sigma_S_t

        # store the results
        dR[i] = dR_t  # realized return from t-1 to t
        r[i] = r_t
        theta[i] = theta_t
        Delta_bar[i] = Delta_bar_t
        mu_S[i] = mu_S_t
        f_mat[i] = f_st
        Delta_mat[i] = Delta_st
        pi_mat[i] = pi_st

    return (
        dR,
        r,
        theta,
        Delta_bar,
        mu_S,
        f_mat,
        Delta_mat,
        pi_mat,
    )

