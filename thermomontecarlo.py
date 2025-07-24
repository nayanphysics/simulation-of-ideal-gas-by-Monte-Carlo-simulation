import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
# Simulation parameters
N = 500              # number of particles
L = 10.0             # size of 3D box (L x L x L)
radius = 0.1         # radius of particles
mass = 1.0           # mass of each particle
dt = 0.005           # time step
steps = 5000         # number of steps
k_B = sc.k           # Boltzmann constant

#user input
'''N=int(input("Enter the no of particles :"))
L= float(input("Enter the length of the edge ofthe cube"))
steps=int(input("Enter  the no of steps))'''

# Initialize particle positions (not overlapping)
positions = np.random.rand(N, 3) * (L - 2 * radius) + radius

# Initialize velocities (random speed and direction)
angles_theta = np.random.rand(N) * 2 * np.pi
angles_phi = np.arccos(2 * np.random.rand(N) - 1)
speeds = np.random.rand(N) * 2

vx = speeds * np.sin(angles_phi) * np.cos(angles_theta)
vy = speeds * np.sin(angles_phi) * np.sin(angles_theta)
vz = speeds * np.cos(angles_phi)
velocities = np.column_stack((vx, vy, vz))

# For pressure calculation
wall_hits = 0
total_momentum = 0

# Track kinetic energy over time
temperature_time = []

def compute_temperature(velocities):
    KE = 0.5 * mass * np.sum(velocities**2)
    T = (2/3) * KE / (N * k_B)   # from equipartition in 3D
    return T, KE

# Run the simulation
for step in range(steps):
    positions += velocities * dt

    for i in range(N):
        for d in range(3):  # x, y, z dimensions
            if positions[i][d] <= radius:
                velocities[i][d] *= -1
                positions[i][d] = radius
                wall_hits += 1
                total_momentum += 2 * mass * abs(velocities[i][d])
            elif positions[i][d] >= L - radius:
                velocities[i][d] *= -1
                positions[i][d] = L - radius
                wall_hits += 1
                total_momentum += 2 * mass * abs(velocities[i][d])

    T, KE = compute_temperature(velocities)
    temperature_time.append(T)

# Calculate pressure: Force/area = momentum / (area * time)
surface_area = 6 * L**2
total_time = steps * dt
pressure = total_momentum / (surface_area * total_time)

# Display results
print(f"Average Temperature: {np.mean(temperature_time):.2f}")
print(f"Estimated Pressure: {pressure:.2f}")

# Plot the speed distribution
speeds = np.linalg.norm(velocities, axis=1)
plt.hist(speeds, bins=25, density=True, alpha=0.7, label="Simulated")

# Maxwell-Boltzmann distribution in 3D
v = np.linspace(0, max(speeds)*1.2, 200)
T_sim = np.mean(temperature_time)
f_v = 4 * np.pi * (mass / (2 * np.pi * k_B * T_sim))**(3/2) * v**2 * np.exp(-mass * v**2 / (2 * k_B * T_sim))
plt.plot(v, f_v, 'r-', label="Maxwell-Boltzmann (3D)")

plt.xlabel("Speed")
plt.ylabel("Probability Density")
plt.title("Maxwell-Boltzmann Speed Distribution (3D)")
plt.legend()
plt.grid(True)
plt.show()
