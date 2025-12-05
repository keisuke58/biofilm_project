# Bayesian Updating of Bacterial Microfilms under Hybrid Uncertainties (Structured Notes)

These notes summarize the repository's interpretation of the paper **"Bayesian updating of bacterial microfilms under hybrid uncertainties with a novel surrogate model"**, with emphasis on **Case II (true two-species submodels for M1 and M2)**. The goals are to keep the implementation aligned with the paper and make the Case II assumptions explicit for development and testing.

## 1. Problem framing
- **Objective:** Calibrate a four-species biofilm growth model by progressively updating parameter groups across three fidelity levels (M1→M2→M3) using TMCMC with a TSM surrogate for fast likelihood evaluation.
- **Quantities of interest:** Volume fractions \(\phi_i\) and porosity factors \(\psi_i\); observed signal per species is \(\phi_i \psi_i\).
- **Uncertainty treatment:** Parametric uncertainty on interaction/growth parameters \(\theta\) and measurement noise; TSM propagates first-order parameter uncertainty to model outputs.

## 2. Model hierarchy (Case II)
- **Species sets:**
  - **M1 (coarse):** Active species 1–2 only (\(a_{11}, a_{12}, a_{22}, b_1, b_2\)).
  - **M2 (medium):** Active species 3–4 only (\(a_{33}, a_{34}, a_{44}, b_3, b_4\)).
  - **M3 (fine):** All four species plus cross-interactions (\(a_{13}, a_{14}, a_{23}, a_{24}\)).
- **Initial states:**
  - **M1:** High-load inoculum for species 1–2 (e.g., \(\phi\_\text{init}=0.2\) per active species); inactive species start from a masked or zeroed state.
  - **M2:** High-load inoculum for species 3–4 (\(\phi\_\text{init}=0.2\) per active species); species 1–2 held inactive.
  - **M3:** Full four-species start-up (\(\phi\_\text{init}=0.02\) per species).
- **Observations:** Sparse time series (typically 20 points) of \(\phi_i \psi_i\) for the active species in each stage.

## 3. Parameter and prior specification
- **Parameter vector:** 14 parameters grouped as above; calibrated in blocks per model stage.
- **Ground-truth values (used for synthetic studies):**
  - Species 1–2: \(a_{11}=0.8, a_{12}=2.0, a_{22}=1.0, b_1=0.1, b_2=0.2\).
  - Species 3–4: \(a_{33}=1.5, a_{34}=1.0, a_{44}=2.0, b_3=0.3, b_4=0.4\).
  - Cross terms: \(a_{13}=2.0, a_{14}=1.0, a_{23}=2.0, a_{24}=1.0\).
- **Priors:** Uniform \([0,3]\) bounds on each parameter.

## 4. Calibration workflow (Case II)
- **Stage 1 (M1):**
  - Fix \(\theta\) components for species 3–4 and cross terms at their reference values.
  - Calibrate \(a_{11}, a_{12}, a_{22}, b_1, b_2\) against \(\phi_1\psi_1, \phi_2\psi_2\).
- **Stage 2 (M2):**
  - Condition on M1 MAP/posterior for species 1–2 and on reference cross terms.
  - Calibrate \(a_{33}, a_{34}, a_{44}, b_3, b_4\) against \(\phi_3\psi_3, \phi_4\psi_4\).
- **Stage 3 (M3):**
  - Condition on M1 and M2 outputs; infer cross-interaction parameters \(a_{13}, a_{14}, a_{23}, a_{24}\) using all four species observables.

## 5. Likelihood and surrogate model
- **Observables:** Mean \(\mu_i = \phi_i \psi_i\); variance propagated via TSM first-order sensitivities: \(\sigma^2(\phi\psi) = \phi^2 \sigma^2(\psi) + \psi^2 \sigma^2(\phi)\).
- **Likelihood:** Heteroscedastic Gaussian with total variance \(\sigma^2 = \sigma^2_{\text{TSM}} + \sigma^2_{\text{obs}}\), typically \(\sigma_{\text{obs}} = 0.005\).
- **TMCMC controls (per paper guidance for sharp posteriors):** target ESS ratio around 0.5–0.8, minimum \(\Delta\beta\) safeguard (~0.01), optional log-likelihood scaling for stability.

## 6. Implementation alignment checklist
- Ensure M1 and M2 solvers honor **true 2-species dynamics** (inactive species masked or zeroed, not merely ignored in outputs).
- Confirm \(\phi\_\text{init}\) handling matches Case II inocula (high initial mass for active species, small/zero for inactive ones without triggering singularities).
- Verify TSM sensitivity bookkeeping respects active-parameter masks so that derivatives are labeled by the parameters actually varied in each stage.
- Keep synthetic data generation (20-point time series of \(\phi_i \psi_i\)) and prior bounds consistent with the above parameterization.
