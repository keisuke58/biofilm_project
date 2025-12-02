# src/numerics.py
import numpy as np
from numba import njit

# Q ベクトル（Numba版）: re_numba.py / biofilm_ultimate.py からそのままコピペ
@njit(cache=True, fastmath=True)
def compute_Q_vector_numba(phi_new, phi0_new, psi_new, gamma_new,
                           phi_old, phi0_old, psi_old,
                           dt, Kp1, Eta_vec, Eta_phi_vec,
                           c_val, alpha_val, A, b_diag):
    Q = np.zeros(10)
    phidot = (phi_new - phi_old) / dt
    phi0dot = (phi0_new - phi0_old) / dt
    psidot = (psi_new - psi_old) / dt
    CapitalPhi = phi_new * psi_new
    Interaction = A @ CapitalPhi
    
    for i in range(4):
        term1 = (Kp1 * (2.0 - 4.0 * phi_new[i])) / ((phi_new[i] - 1.0)**3 * phi_new[i]**3)
        term2 = (1.0 / Eta_vec[i]) * (gamma_new + (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) * phidot[i] +
                Eta_vec[i] * phi_new[i] * psi_new[i] * psidot[i])
        term3 = (c_val / Eta_vec[i]) * psi_new[i] * Interaction[i]
        Q[i] = term1 + term2 - term3
    
    Q[4] = gamma_new + (Kp1 * (2.0 - 4.0 * phi0_new)) / ((phi0_new - 1.0)**3 * phi0_new**3) + phi0dot
    
    for i in range(4):
        term1 = (-2.0 * Kp1) / ((psi_new[i] - 1.0)**2 * psi_new[i]**3) - \
                (2.0 * Kp1) / ((psi_new[i] - 1.0)**3 * psi_new[i]**2)
        term2 = (b_diag[i] * alpha_val / Eta_vec[i]) * psi_new[i]
        term3 = phi_new[i] * psi_new[i] * phidot[i] + phi_new[i]**2 * psidot[i]
        term4 = (c_val / Eta_vec[i]) * phi_new[i] * Interaction[i]
        Q[5+i] = term1 + term2 + term3 - term4
    
    Q[9] = phi_new[0] + phi_new[1] + phi_new[2] + phi_new[3] + phi0_new - 1.0
    return Q


@njit(cache=True, fastmath=True)
def compute_jacobian_numba(phi_new, phi0_new, psi_new, gamma_new,
                           phi_old, psi_old, dt, Kp1,
                           Eta_vec, Eta_phi_vec, c_val, alpha_val, A, b_diag):
    K = np.zeros((10, 10))
    phidot = (phi_new - phi_old) / dt
    psidot = (psi_new - psi_old) / dt
    CapitalPhi = phi_new * psi_new
    Interaction = A @ CapitalPhi
    
    phi_p_deriv = np.zeros(4)
    for i in range(4):
        v = phi_new[i]
        phi_p_deriv[i] = (Kp1*(-4. + 8.*v))/((v**3)*(v-1.)**3) - \
                        (Kp1*(2. - 4.*v))*(3./((v**4)*(v-1.)**3) + 3./((v**3)*(v-1.)**4))
    
    v0 = phi0_new
    phi0_p_deriv = (Kp1*(-4. + 8.*v0))/((v0**3)*(v0-1.)**3) - \
                (Kp1*(2. - 4.*v0))*(3./((v0**4)*(v0-1.)**3) + 3./((v0**3)*(v0-1.)**4))
    
    psi_p_deriv = np.zeros(4)
    for i in range(4):
        v = psi_new[i]
        psi_p_deriv[i] = (4.0 * Kp1 * (3.0 - 5.0*v + 5.0*v**2)) / ((v**4) * (v - 1.0)**4)
    
    for i in range(4):
        for j in range(4):
            K[i, j] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * psi_new[j])
        K[i, i] = phi_p_deriv[i] + (1.0 / Eta_vec[i]) * (
            (Eta_phi_vec[i] + Eta_vec[i] * psi_new[i]**2) / dt +
            Eta_vec[i] * psi_new[i] * psidot[i]) - \
            (c_val / Eta_vec[i]) * (psi_new[i] * (Interaction[i] + A[i, i] * psi_new[i]))
        K[i, 4] = 0.0
        for j in range(4):
            K[i, j+5] = (c_val / Eta_vec[i]) * psi_new[i] * (-A[i, j] * phi_new[j])
        K[i, i+5] = (1.0 / Eta_vec[i]) * (
            2.0 * Eta_vec[i] * psi_new[i] * phidot[i] +
            Eta_vec[i] * phi_new[i] * psidot[i] +
            Eta_vec[i] * phi_new[i] * psi_new[i] / dt) - \
            (c_val / Eta_vec[i]) * ((Interaction[i] + A[i, i] * phi_new[i] * psi_new[i]) +
                                    psi_new[i] * (A[i, i] * phi_new[i]))
        K[i, 9] = 1.0 / Eta_vec[i]
    
    K[4, 4] = phi0_p_deriv + 1.0/dt
    K[4, 9] = 1.0
    
    for i in range(4):
        k = i + 5
        for j in range(4):
            K[k, j] = -(c_val / Eta_vec[i]) * (A[i, j] * psi_new[j] * phi_new[i] +
                    Interaction[i] * (1.0 if i == j else 0.0))
        K[k, i] = (psi_new[i] * phidot[i] + psi_new[i] * phi_new[i] / dt +
                2.0 * phi_new[i] * psidot[i]) - \
                (c_val / Eta_vec[i]) * (A[i, i] * psi_new[i] * phi_new[i] +
                                        Interaction[i] + phi_new[i] * A[i, i] * psi_new[i])
        K[k, 4] = 0.0
        for j in range(4):
            K[k, j+5] = -(c_val / Eta_vec[i]) * phi_new[i] * A[i, j] * phi_new[j]
        K[k, i+5] = psi_p_deriv[i] + (b_diag[i] * alpha_val / Eta_vec[i]) + \
                    (phi_new[i] * phidot[i] + phi_new[i]**2 / dt) - \
                    (c_val / Eta_vec[i]) * phi_new[i] * A[i, i] * phi_new[i]
        K[k, 9] = 0.0
    
    K[9, 0] = 1.0
    K[9, 1] = 1.0
    K[9, 2] = 1.0
    K[9, 3] = 1.0
    K[9, 4] = 1.0
    return K

@njit(cache=True, fastmath=True)
def dQ_dtheta_analytical_numba(phi_new, psi_new, c_val, alpha_val, Eta_vec,
                               CapitalPhi, theta_idx):
    """
    ANALYTICAL SENSITIVITY: Compute ∂Q/∂θ_k exactly
    
    θ = [a11, a12, a22, b1, b2, a33, a34, a44, b3, b4, a13, a14, a23, a24]
            0     1    2    3   4   5    6    7    8   9   10   11   12   13
    
    A matrix structure (symmetric):
    A[0,0]=a11, A[0,1]=A[1,0]=a12, A[0,2]=A[2,0]=a13, A[0,3]=A[3,0]=a14
    A[1,1]=a22, A[1,2]=A[2,1]=a23, A[1,3]=A[3,1]=a24
    A[2,2]=a33, A[2,3]=A[3,2]=a34
    A[3,3]=a44
    
    b = [b1, b2, b3, b4] at indices [3, 4, 8, 9]
    """
    dQ = np.zeros(10)
    
    # Mapping: theta_idx -> (row, col) in A, or b_index
    # Diagonal A elements: affect only one (i,i) entry
    # Off-diagonal A elements: affect both (i,j) and (j,i) due to symmetry
    
    if theta_idx == 0:  # a11 -> A[0,0]
        # Q[0]: -c/η_0 * ψ_0 * (∂A@Φ)_0 = -c/η_0 * ψ_0 * Φ_0
        dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[0]
        # Q[5]: -c/η_0 * φ_0 * (∂A@Φ)_0 = -c/η_0 * φ_0 * Φ_0
        dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[0]
        
    elif theta_idx == 1:  # a12 -> A[0,1] and A[1,0]
        # Affects row 0: ∂(A@Φ)_0/∂a12 = Φ_1
        # Affects row 1: ∂(A@Φ)_1/∂a12 = Φ_0
        dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[1]
        dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[0]
        dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[1]
        dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[0]
        
    elif theta_idx == 2:  # a22 -> A[1,1]
        dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[1]
        dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[1]
        
    elif theta_idx == 3:  # b1
        # Q[5]: (b1 * α / η_0) * ψ_0 -> ∂/∂b1 = α/η_0 * ψ_0
        dQ[5] = (alpha_val / Eta_vec[0]) * psi_new[0]
        
    elif theta_idx == 4:  # b2
        dQ[6] = (alpha_val / Eta_vec[1]) * psi_new[1]
        
    elif theta_idx == 5:  # a33 -> A[2,2]
        dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[2]
        dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[2]
        
    elif theta_idx == 6:  # a34 -> A[2,3] and A[3,2]
        dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[3]
        dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[2]
        dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[3]
        dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[2]
        
    elif theta_idx == 7:  # a44 -> A[3,3]
        dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[3]
        dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[3]
        
    elif theta_idx == 8:  # b3
        dQ[7] = (alpha_val / Eta_vec[2]) * psi_new[2]
        
    elif theta_idx == 9:  # b4
        dQ[8] = (alpha_val / Eta_vec[3]) * psi_new[3]
        
    elif theta_idx == 10:  # a13 -> A[0,2] and A[2,0]
        dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[2]
        dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[0]
        dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[2]
        dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[0]
        
    elif theta_idx == 11:  # a14 -> A[0,3] and A[3,0]
        dQ[0] = -(c_val / Eta_vec[0]) * psi_new[0] * CapitalPhi[3]
        dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[0]
        dQ[5] = -(c_val / Eta_vec[0]) * phi_new[0] * CapitalPhi[3]
        dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[0]
        
    elif theta_idx == 12:  # a23 -> A[1,2] and A[2,1]
        dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[2]
        dQ[2] = -(c_val / Eta_vec[2]) * psi_new[2] * CapitalPhi[1]
        dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[2]
        dQ[7] = -(c_val / Eta_vec[2]) * phi_new[2] * CapitalPhi[1]
        
    elif theta_idx == 13:  # a24 -> A[1,3] and A[3,1]
        dQ[1] = -(c_val / Eta_vec[1]) * psi_new[1] * CapitalPhi[3]
        dQ[3] = -(c_val / Eta_vec[3]) * psi_new[3] * CapitalPhi[1]
        dQ[6] = -(c_val / Eta_vec[1]) * phi_new[1] * CapitalPhi[3]
        dQ[8] = -(c_val / Eta_vec[3]) * phi_new[3] * CapitalPhi[1]
    
    return dQ

@njit(cache=True, fastmath=True)
def sigma2_accumulate_numba(x1, var_theta_active):
    """Accumulate variance: σ² = Σ_k (x1[:,:,k])² * Var(θ_k)"""
    n_time, n_state, n_theta = x1.shape
    sigma2 = np.zeros((n_time, n_state)) + 1e-12
    for k in range(n_theta):
        for t in range(n_time):
            for s in range(n_state):
                sigma2[t, s] += (x1[t, s, k]**2) * var_theta_active[k]
    return sigma2
