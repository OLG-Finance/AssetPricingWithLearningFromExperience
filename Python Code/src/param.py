import numpy as np

# Parameters
rho = 0.001  # Time discount factor
nu = 0.02  # Death rate
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_S = np.copy(sigma_Y)  # In equilibrium the stock price diffusion is the same as output diffusion
w = 0.92  # Fraction of total output paid out as endowment

# Some pre-calculations
D = rho ** 2 + 4 * (rho * nu + nu ** 2) * (1 - w)
beta = (rho + 2 * nu - D ** 0.5) / (2 * nu)
r_log = rho + mu_Y - sigma_Y ** 2  # interest rate in a representative agent economy with log utility

# Setting prior variance
T_hat = 20  # Pre-trading years
dt = 1 / 12  # time incremental
N_pre = int(T_hat / dt)  # Pre-trading periods
V_hat = (sigma_Y ** 2) / T_hat  # prior variance
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)
Nc = np.copy(Nt)
tau = np.arange(T_cohort, 0, -dt)


