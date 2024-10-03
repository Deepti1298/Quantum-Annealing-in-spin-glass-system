import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d

def btest(a, i):
    return (a >> i) & 1

def construct_hamiltonian(g, h, Γ0):
    N = len(g)
    config = 2**N
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
                H[a, flip_i] -= Γ0
                H[flip_i, a] -= Γ0

    return H

def calculate_local_order_parameters(ground_state, N):
    Q_i = np.zeros(N)

    for i in range(N):
        sigma_z_i = np.zeros((2**N, 2**N))
        for a in range(2**N):
            if btest(a, i):
                sigma_z_i[a, a] = 1
            else:
                sigma_z_i[a, a] = -1
        Q_i[i] = np.dot(ground_state.conj().T, np.dot(sigma_z_i, ground_state))**2

    return np.abs(Q_i)

Γ0 = 0.3  
N = 8
h_values = [ 0.1]
num_realizations = 16
bins = 200  

combined_data = []


plt.figure(figsize=(12, 8))

for h in h_values:
    all_Q = []

    for realization in range(num_realizations):
        g = np.random.rand(N, N)
        H = construct_hamiltonian(g, h, Γ0)
        eigenvalues, eigenstates = eigh(H)
        ground_state = eigenstates[:, 0]
        
        
        Q_i = calculate_local_order_parameters(ground_state, N)
        all_Q.extend(Q_i)

    
    Q_vals = np.linspace(0, 1, bins)
    hist, _ = np.histogram(all_Q, bins=Q_vals, density=True)  

    
    hist_smoothed = gaussian_filter1d(hist, sigma=2)

    
    for i in range(len(Q_vals) - 1):
        combined_data.append([Q_vals[i], h, hist_smoothed[i]])

    
    plt.plot(Q_vals[:-1], hist_smoothed, label=f"h={h}", marker='o', linestyle='-', markersize=5)



for h in h_values:
    filename = f"Q_values_per_realization_N{N}_h{h}_16_realizations.txt"
    with open(filename, "w") as f:
        for realization, Q_vals in enumerate(Q_values_per_realization[h]):
            f.write(f"Realization {realization + 1} Q values:\n")
            f.write(f"{Q_vals}\n\n")
    print(f"Q values for h={h} saved to {filename}")


plt.title("Smoothed Distribution of Local Order Parameters |Q| for N=8 (Averaged over 12 Realizations)")
plt.xlabel("|Q|")
plt.ylabel("P(|Q|)")
plt.legend()
plt.grid(True)
plt.show()


