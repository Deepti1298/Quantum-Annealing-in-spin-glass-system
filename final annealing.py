import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import csv
from numba import njit

N = 8
config = 2**N
Γ0 = 4.0
h = 0.1
τ = 1000  
num_realizations = 5
dt = 0.1


with open('finre.txt', 'w') as file:
    
    def btest(a, i):
        return (a >> i) & 1

    def hamiltonian(t, g, h):
        Γ_t = Γ0 * (1 - t / τ)
        H = np.zeros((config, config), dtype=float)
        for a in range(config):
            for i in range(N):
                for j in range(N):
                    if i < j:
                        if btest(a, i) == btest(a, j):
                            H[a, a] -= g[i, j]
                        else:
                            H[a, a] += g[i, j]
            for i in range(N):
                if btest(a, i):
                    H[a, a] -= h
                else:
                    H[a, a] += h
            for i in range(N):
                flip_i = a ^ (1 << i)
                if flip_i < config:
                    H[a, flip_i] -= Γ_t
                    H[flip_i, a] -= Γ_t
        return H

    def classical_Hamiltonian(g):
        H_classical = np.zeros((config, config), dtype=float)
        for a in range(config):
            for i in range(N):
                for j in range(N):
                    if i < j:
                        if btest(a, i) == btest(a, j):
                            H_classical[a, a] -= g[i, j]
                        else:
                            H_classical[a, a] += g[i, j]
        return H_classical

    def get_classical_ground_state(H_classical):
        eigenvalues, eigenvectors = eigh(H_classical)
        eigen_value = eigenvalues[0]
        psi_classical = eigenvectors[:, 0]
        return eigen_value, psi_classical

    def initial_state(g, h):
        H0 = hamiltonian(0, g, h)
        eigenvalues, eigenvectors = eigh(H0)
        psi_0 = eigenvectors[:, 0]
        return psi_0

    def time_dependent_schrodinger(t, psi, g, h):
        H_t = hamiltonian(t, g, h)
        dpsi_dt = -1j * np.dot(H_t, psi)
        return dpsi_dt

    def runge_kutta(psi_0, t0, tf, dt, g, h):
        t = t0
        psi = psi_0.astype(complex)
        while t < tf:
            k1 = dt * time_dependent_schrodinger(t, psi, g, h)
            k2 = dt * time_dependent_schrodinger(t + dt / 2, psi + k1*dt / 2, g, h)
            k3 = dt * time_dependent_schrodinger(t + dt / 2, psi + k2*dt/ 2, g, h)
            k4 = dt * time_dependent_schrodinger(t + dt, psi + k3*dt, g, h)
            psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += dt
        return psi

    def instantaneous_probability(psi_classical, psi_t):
        norm_psi_t = np.linalg.norm(psi_t)
        if norm_psi_t > 0:
            psi_t /= norm_psi_t
        return (np.abs(np.dot(np.conj(psi_classical), psi_t)))**2

    t0 = 0
    tf = τ

    time_points = np.arange(t0, tf, dt)
    all_probabilities = []

    for realization in range(num_realizations):
        J = 1.0
        g = np.random.randn(N, N) * J / np.sqrt(N)
        g = (g + g.T) / 2
        H_classical = classical_Hamiltonian(g)
        eigen_value, psi_classical = get_classical_ground_state(H_classical)
        psi_0 = initial_state(g, h)

        probabilities = []
        for t in time_points:
            psi_t = runge_kutta(psi_0, t, t + dt, dt, g, h)
            prob = instantaneous_probability(psi_classical, psi_t)
            probabilities.append(prob)
            psi_0 = psi_t  

        all_probabilities.append(probabilities)
        type(all_probabilities)
        prob_all=np.array(all_probabilities)
        np.shape(prob_all)
        len(all_probabilities)

    for i, t in enumerate(time_points):
        probs_str = ', '.join(f'{prob:.5f}' for prob in [realization_prob[i] for realization_prob in all_probabilities])
        file.write(f'Time: {t:.2f}, Probabilities: {probs_str}\n')


plt.figure(figsize=(10, 6))
for realization_prob in all_probabilities:
    plt.plot(time_points, realization_prob)
    
plt.xlabel('Time')
plt.ylabel('Probability of Finding Ground State')
plt.title('Time-evolution of Ground State Probability with Constant Longitudinal Field h=0.1')
plt.grid(True)
plt.show()


