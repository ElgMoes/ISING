import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import deque
from icecream import ic  # debugging tool
from tqdm import tqdm

begin_time = datetime.now()

# Set NumPy to ignore overflow warnings
np.seterr(over='ignore')

"""Constants and setup"""
np.random.seed(42)
num_seeds = 5
seeds = np.random.randint(1, 10**9, size=num_seeds).tolist()
n = 100  # Size of the lattice (n x n)
loops = 1000  # Number of flipping attempts
wolff_loops_devider = 2 # wollf algorithm runs loops/wolff_loops_devider times instead of loops
T = [.5, 1, 2, 3, 4, 5, 8, 10] # reduced temperature
simulations = len(T)
last=500

# Define algorithm names for display
algorithm_names = ['Metropolis', 'Wolff']

"""setting up arrays to store desired variables"""
mean_energy_per_sim = np.zeros(shape = (len(seeds), 2, simulations), dtype=int)
mean_abs_mean_magnetisation = np.ones(shape = (len(seeds), 2, simulations))
heat_capacitance = np.zeros(shape = (len(seeds), 2, simulations), dtype=int)
energy_array = np.zeros(shape = (len(seeds), 2, simulations, loops), dtype=int)
mean_magnetisation_array = np.zeros(shape=(len(seeds), 2, simulations, loops), dtype=int)
time_array = np.linspace(0, loops, loops)

def calculate_system_energy(arr): # both @metropolis and @wolff
    """Calculates the total system energy"""
    # Nearest neighbor shifts (using periodic boundary conditions with np.roll)
    right = np.roll(arr, -1, axis=1)
    left = np.roll(arr, 1, axis=1)
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)

    # Calculate energy by summing interactions with neighbours
    E_sum = -np.sum(arr * (right + left + up + down))
    return E_sum

def calculate_difference_energy(arr, i, j):
    """optimized way to calculate the change in energy when a flip is attempted @metropolis"""
    (n, m) = np.shape(arr)
    neighbors_i = [(i - 1) % n, i, (i + 1) % n]
    neighbors_j = [(j - 1) % m, j, (j + 1) % m]

    neighbor_array = arr[np.ix_(neighbors_i, neighbors_j)] # creating an array of all the points around the flipped object

    E_before = calculate_system_energy(neighbor_array)
    temp_neighbor_arr = flip_spin(neighbor_array, 1, 1)
    E_after = calculate_system_energy(temp_neighbor_arr)
    temp_arr = flip_spin(arr, i, j)
    dE = E_after - E_before
    return temp_arr, dE

def flip_spin(arr, i, j):
    """function to invert the value of (i, j) from an array @metropolis"""
    arr[i][j] *= -1
    return arr

def try_flip(seed, sim, loop, arr, i, j, T):
    """attempts to flip an object and flips it if possible @metropolis"""
    old_arr = arr.copy() # copies the data from the inserted array to return when a flip fails

    temp_arr, dE = calculate_difference_energy(arr, i, j)
    p = np.exp(-dE/(T))
    R = np.random.random()

    global energy_array

    if (dE < 0 or R < p): # flip is benificial for energy or the random chance flips the object
        global energy
        energy += dE
        energy_array[seed][0][sim][loop] = energy

        return temp_arr
    else: # flipping the object failed
        energy_array[seed][0][sim][loop] = energy
        return old_arr

def metropolis(seed, sim, loop):
    """"""
    # getting random coordinates to flip
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    global spin_lattice # global is needed to access the variable generated later in the script
    spin_lattice = try_flip(seed, sim, loop, spin_lattice, i, j, T[sim])

    # calculating mean magnetisation and adding it to the array
    mean_magnetisation = np.absolute(np.mean(spin_lattice))
    global mean_magnetisation_array # generated later in script
    mean_magnetisation_array[seed][0][sim][loop] = mean_magnetisation

    return spin_lattice

def flip_wolff_cluster(seed, sim, loop, lattice, i, j, T):
    """Main flipping function, creates and flips clusers"""
    old_lattice = lattice.copy() # saving it for when flip fails
    initial_spin = lattice[i, j]
    lattice[i, j] *= -1 # flips first object
    p_add = 1 - np.exp(-2 / T) # generates possibility of neighbouring object to be added to cluster
    queue = deque([(i, j)])  # Queue of cluster coordinates
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    (n, m) = np.shape(lattice)

    # loops over stack to check for new possible cluster members
    while queue:
        (x, y) = queue.popleft()
        for dx, dy in neighbours:
            nx, ny = (x + dx) % n, (y + dy) % m # creating new coordinates to be added to the cluster
            if lattice[nx, ny] == initial_spin and np.random.random() < p_add: # checks if new coordinate has the same spin and is below the probability threshold
                lattice[nx, ny] *= 0 # flips the newly added object
                queue.append((nx, ny)) # adds the new object to the cluster stack

    lattice[lattice == 0] = -1*initial_spin

    # calculating energy difference after cluster flip
    E_before = calculate_system_energy(old_lattice)
    E_after = calculate_system_energy(lattice)
    dE = E_after - E_before

    # generates possibility of flip with positive dE
    p = np.exp(-dE / T) if T > 0 else 0
    R = np.random.random()

    global energy_array # some variables are declared after the function, so they need to be global if accessed from within the function

    if dE <= 0 or R < p:
        global energy
        energy += dE
        energy_array[seed][1][sim][loop] = energy
        return lattice # returns new lattice with flipped cluster
    else:
        energy_array[seed][1][sim][loop] = energy
        return old_lattice # returns old lattice when cluster flip failed
    
def wolff_step(seed, sim, loop, lattice, T):
    """Loops over the cluster creation and flipping"""
    T = T[sim]
    i, j = np.random.randint(0, n, size=2) # generates random coordinates
    lattice = flip_wolff_cluster(seed, sim, loop, lattice, i, j, T)

    # storing magnetisation over every step
    global mean_magnetisation_array
    mean_magnetisation_array[seed][1][sim][loop] = np.absolute(np.mean(lattice))
    return lattice

def initialize_spin_lattice(n):
    """Initialize the spin lattice with random spins."""
    return np.random.choice([1, -1], size=(n, n))

def process_simulation(seed, sim, algorithm, spin_lattice, loops, last, T, energy_array, mean_magnetisation_array):
    """Process a simulation using the specified algorithm (Metropolis or Wolff)."""
    # Calculate initial energy and magnetization
    global energy
    energy = calculate_system_energy(spin_lattice)
    energy_array[seed][algorithm][sim][0] = energy
    mean_magnetisation_array[seed][algorithm][sim][0] = np.mean(spin_lattice)

    if algorithm == 0:  # Metropolis
        for loop in tqdm(range(loops), desc=f"Processing Steps", leave=False):
            spin_lattice = metropolis(seed, sim, loop)
    elif algorithm == 1:  # Wolff
        for loop in tqdm(range(500 if T[sim]<0 else loops), desc=f"Processing Steps", leave=False):
            spin_lattice = wolff_step(seed, sim, loop, spin_lattice, T)
        if T[sim]<0:
            energy_array[500:loops] = -4*n*n

    # Getting averages of desired variables
    mean_energy_per_sim[seed][algorithm][sim] = np.mean(energy_array[seed][algorithm][sim][-last:])
    mean_abs_mean_magnetisation[seed][algorithm][sim] = np.mean(np.abs(mean_magnetisation_array[seed][algorithm][sim][-last:]))
    
    # Heat capacity calculations
    heat_capacitance[seed][algorithm][sim] = np.var(energy_array[seed][algorithm][sim][-last:]) / T[sim]

# Main processing loop
with tqdm(total=len(seeds)*2*simulations, desc="Simulation") as pbar:
    for seed in range(len(seeds)):
        np.random.seed(seeds[seed])  # Set seed for reproducibility
        for sim in range(simulations):
            for algorithm in range(2):  # 0 for Metropolis, 1 for Wolff
                spin_lattice = initialize_spin_lattice(n)  # Initialize spin lattice
                process_simulation(seed, sim, algorithm, spin_lattice, loops, last, T, energy_array, mean_magnetisation_array)
                pbar.update(1)


# setting up data figures and plots
fig_graph, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
fig_dots, (ax5, ax6, ax7) = plt.subplots(ncols=1, nrows=3, figsize=(16, 10))
for seed in range(0, len(seeds)):
    for sim in range(simulations):
        ax1.plot(time_array, energy_array[seed][0][sim], linewidth=0.5)
        ax2.plot(time_array, mean_magnetisation_array[seed][0][sim], linewidth=0.5)
        ax3.plot(time_array, energy_array[seed][1][sim], linewidth=0.5)
        ax4.plot(time_array, mean_magnetisation_array[seed][1][sim], linewidth=0.5)
    ax1.set_xlabel("Time")
    ax2.set_xlabel("Time")
    ax3.set_xlabel("Time")
    ax4.set_xlabel("Time")
    ax1.set_ylabel("Energy")
    ax2.set_ylabel("Mean absolute magnetisation")
    ax3.set_ylabel("Energy")
    ax4.set_ylabel("Mean absolute magnetisation")
    ax1.set_title("Energy of Metropolis algorithm")
    ax2.set_title("Mean absolute magnetisation of Metropolis algorithm")
    ax3.set_title("Energy of Wolff algorithm")
    ax4.set_title("Mean absolute magnetisation of Wolff algorithm")

def plot_with_error(ax, T, means, stds=None, label_prefix=''):
    # Scatter plot
    ax.scatter(T, means, label=f'{label_prefix} Mean')
    
    # Error bars if stds are provided
    if stds is not None:
        ax.errorbar(T, means, yerr=stds, fmt='o', label=f'{label_prefix} Error Bars')

# Means for energy, magnetization, and heat capacity
mean_energy_m = np.mean(mean_energy_per_sim[:, 0], axis=0)
mean_energy_w = np.mean(mean_energy_per_sim[:, 1], axis=0)
mean_magnetisation_m = np.mean(mean_abs_mean_magnetisation[:, 0], axis=0)
mean_magnetisation_w = np.mean(mean_abs_mean_magnetisation[:, 1], axis=0)
mean_heat_capacitance_m = np.mean(heat_capacitance[:, 0], axis=0)
mean_heat_capacitance_w = np.mean(heat_capacitance[:, 1], axis=0)

# Standard deviations
std_energy_m = np.std(mean_energy_per_sim[:, 0], axis=0)
std_energy_w = np.std(mean_energy_per_sim[:, 1], axis=0)
std_magnetisation_m = np.std(mean_abs_mean_magnetisation[:, 0], axis=0)
std_magnetisation_w = np.std(mean_abs_mean_magnetisation[:, 1], axis=0)
std_heat_capacitance_m = np.std(heat_capacitance[:, 0], axis=0)
std_heat_capacitance_w = np.std(heat_capacitance[:, 1], axis=0)

# Plotting Energy
plot_with_error(ax5, T, mean_energy_m, std_energy_m, label_prefix='Metropolis')  # First energy mean
plot_with_error(ax5, T, mean_energy_w, std_energy_w, label_prefix='Wolff')  # Second energy mean
ax5.set_xlabel("reduced temperature")
ax5.set_ylabel("Energy")

# Plotting Mean Absolute Magnetisation
plot_with_error(ax6, T, mean_magnetisation_m, std_magnetisation_m, label_prefix='Metropolis')  # First magnetisation mean
plot_with_error(ax6, T, mean_magnetisation_w, std_magnetisation_w, label_prefix='Wolff')  # Second magnetisation mean
ax6.set_xlabel("reduced temperature")
ax6.set_ylabel("Mean absolute magnetisation")

# Plotting Heat Capacity
plot_with_error(ax7, T, mean_heat_capacitance_m, std_heat_capacitance_m, label_prefix='Metropolis')  # First heat capacity mean
plot_with_error(ax7, T, mean_heat_capacitance_w, std_heat_capacitance_w, label_prefix='Wolff')  # Second heat capacity mean
ax7.set_xlabel("reduced temperature")
ax7.set_ylabel("Heat capacity")

plt.tight_layout()
fig_graph.savefig("ising_graph_comparison.png")
fig_dots.savefig("ising_data.png")

end_time = datetime.now()
total_time = end_time-begin_time
print(total_time)