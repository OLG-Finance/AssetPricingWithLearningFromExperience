from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from src.param import nu, mu_Y, sigma_Y, sigma_S, beta, T_hat, dt, N_pre, V_hat, Nt, Nc, tau
from src.functions import build_cohorts, simulate_cohorts

Mpath = 100
dZ_build_mat = np.random.randn(Mpath, Nc) * np.sqrt(dt)
dZ_mat = np.random.randn(Mpath, Nt) * np.sqrt(dt)

# initialize the matrices for storage
dR_store = np.zeros((Mpath, Nt), dtype=np.float32)
r_store = np.zeros((Mpath, Nt), dtype=np.float32)
theta_store = np.zeros((Mpath, Nt), dtype=np.float32)
Delta_bar_store = np.zeros((Mpath, Nt), dtype=np.float32)
mu_S_store = np.zeros((Mpath, Nt), dtype=np.float32)

age_sample = np.arange(0, 100, 5)
f_mat_store = np.zeros((Mpath, Nt, len(age_sample)), dtype=np.float32)
Delta_mat_store = np.zeros((Mpath, Nt, len(age_sample)), dtype=np.float32)
pi_mat_store = np.zeros((Mpath, Nt, len(age_sample)), dtype=np.float32)

for i in tqdm(range(Mpath)):
    dZ_build = dZ_build_mat[i]
    dZ = dZ_mat[i]
    bias_vec = dZ_build[-N_pre:]

    (
        Delta_st,
        eta_st_eta_ss,
        X,
    ) = build_cohorts(
        dZ_build,
        Nc,
        dt,
        tau,
        nu,
        beta,
        V_hat,
        sigma_Y,
        N_pre,
    )

    (
        dR,
        r,
        theta,
        Delta_bar,
        mu_S,
        f_mat,
        Delta_mat,
        pi_mat,
    ) = simulate_cohorts(
        bias_vec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        nu,
        beta,
        V_hat,
        mu_Y,
        sigma_Y,
        T_hat,
        N_pre,
        eta_st_eta_ss,
        X,
        Delta_st
    )

    dR_store[i] = dR
    r_store[i] = r
    theta_store[i] = theta
    Delta_bar_store[i] = Delta_bar
    mu_S_store[i] = mu_S

    # save only a sample to save storage space
    f_mat_store[i] = f_mat[:, -age_sample*12-1]
    Delta_mat_store[i] = Delta_mat[:, -age_sample*12-1]
    pi_mat_store[i] = pi_mat[:, -age_sample*12-1]


# visualization: Example
# correlation between true and perceived risk premium given age:

mu_S_r = theta_store * sigma_S
ave_corr_mu = np.zeros(len(age_sample))
for i in range(len(age_sample)):
    mu_S_r_st = (theta_store + Delta_mat_store[:, :, i]) * sigma_S
    ave_corr_mu[i] = np.corrcoef(np.reshape(mu_S_r, -1), np.reshape(mu_S_r_st, -1))[1, 0]

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(age_sample, ave_corr_mu, linewidth=2)
plt.savefig('correlation_real_perceived.png', dpi=100)
plt.show()
plt.close()






