This file replicates the results from "Asset Prices and Portfolio Choice with Learning from Experience" by Ehling, Graniero, and Heyerdahl-Larsen (2017). 

The code is rewritten from the Matlab code.

For any questions, please contact: zeshu.xu@bi.no.

The file contains three python py files:
- **`main.py`**: does the simulation with one example figure. 
- **`params.py`**: contains the parameters used in the simulations.
- **`functions.py`**: defines the main functions, including:
  - **`post_var`**: updates V
  - **`dDelta_st_calculator`**: updates Delta_st
  - **`build_cohorts`**: builds up a large number of cohorts for an OLG economy
  - **`simulate_cohorts`**: simulate the OLG economy forward, keeping the number of cohorts constant