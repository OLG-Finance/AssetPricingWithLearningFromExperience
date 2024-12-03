# Matlab Code

This file replicates the results from "Asset Prices and Portfolio Choice with Learning from Experience" by Ehling, Graniero, and Heyerdahl-Larsen (2017). The code can also be downloaded at ReStud's website. 

For any questions, please contact: [alessandro.graniero@bi.no](mailto:alessandro.graniero@bi.no).  

The file contains five Matlab M-files:

- **`MainFile_RES.m`**: This is the main file to replicate the full set of results of Section 3 in the paper, namely Figure 1, Figure 2, Figure 3, and Table 1.
- **`PostVar.m`**, **`BuildUpCohortsMAIN.m`**, **`SimCohortsMAIN.m`**, and **`olsgmm.m`**: These are m-files called in by the main file `MainFile_RES.m`.
  - **`PostVar.m`**: This function calculates the posterior variance of the Kalman filter.
  - **`BuildUpCohortsMAIN.m`**: This creates the stationary economy with a large number of cohorts.
  - **`SimCohortsMAIN.m`**: This simulates the economy forward. The number of simulated paths in the code is set to a default value of 100. The results reported in the paper are based on 10,000 paths.
  - **`olsgmm.m`**: This function performs the OLS regressions with GMM-corrected standard errors. The results of Table 1 are obtained using this function.

Make sure to include all the above files in the same directory to replicate the results of the paper.
