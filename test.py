import numpy as np
from icecream import ic  # debugging tool
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm


# Constants and setup
seeds=[873193017]
n = 20  # Size of the lattice (n x n)
loops = 20000  # Number of flipping attempts
T = [1, 2, 3, 4, 5, 6, 8, 10] # effective temperature
simulations = len(T)
last=1000

def calculate_system_energy(arr): # both metropolis and wolff
    # Nearest neighbor shifts (using periodic boundary conditions with np.roll)
    right = np.roll(arr, -1, axis=1)
    left = np.roll(arr, 1, axis=1)
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)

    # Calculate energy by summing interactions with neighbours
    E_sum = -np.sum(arr * (right + left + up + down))
    return E_sum

def calculate_difference_energy(arr, i, j): # optimized way to calculate the change in energy when a flip is attempted # metropolis
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

def flip_spin(arr, i, j): # function to invert the value of (i, j) from an array # metropolis
    arr[i][j] *= -1
    return arr

def try_flip(arr, i, j, T, sim, loop, seed): # attempts to flip an object and flips it if possible # metropolis
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

def metropolis(sim=1, loop=0, seed=0):
    # getting random coordinates to flip
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    global spin_lattice # global is needed to access the variable generated later in the script
    spin_lattice = try_flip(spin_lattice, i, j, T[sim], sim, loop, seed)

    # calculating mean magnetisation and adding it to the array
    mean_magnetisation = np.absolute(np.mean(spin_lattice))
    global mean_magnetisation_array # generated later in script
    mean_magnetisation_array[seed][0][sim][loop] = mean_magnetisation

    return spin_lattice

# main flipping function, creates and flips clusers
def flip_wolff_cluster(lattice, i, j, T, sim=1, loop=1, seed=0):
    old_lattice = lattice.copy() # saving it for when flip fails
    initial_spin = lattice[i, j]
    lattice[i, j] *= -1 # flips first object
    p_add = 1 - np.exp(-2 / T) # generates possibility of neighbouring object to be added to cluster
    stack = [(i, j)] # stack of cluster coordinates
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    (n, m) = np.shape(lattice)

    # loops over stack to check for new possible cluster members
    while stack:
        (x, y) = stack.pop()
        for dx, dy in neighbours:
            nx, ny = (x + dx) % n, (y + dy) % m # creating new coordinates to be added to the cluster
            if lattice[nx, ny] == initial_spin and np.random.random() < p_add: # checks if new coordinate has the same spin and is below the probability threshold
                lattice[nx, ny] *= -1 # flips the newly added object
                stack.append((nx, ny)) # adds the new object to the cluster stack

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
    
# loops over the cluster creation and flipping
def wolff_step(lattice, T, sim=1, loop=1, seed=0):
    T = T[sim]
    i, j = np.random.randint(0, n, size=2) # generates random coordinates
    lattice = flip_wolff_cluster(lattice, i, j, T, sim, loop, seed)

    # storing magnetisation over every step
    global mean_magnetisation_array
    mean_magnetisation_array[seed][1][sim][loop] = np.absolute(np.mean(lattice))
    return lattice

# setting up arrays to store desired variables
mean_energy_per_sim = np.zeros(shape = (len(seeds), 2, simulations))
mean_energy_per_sim_last = np.zeros(shape = (len(seeds), 2, simulations))
mean_energy_per_sim_last_std = np.zeros(shape = (len(seeds), 2, simulations))
mean_abs_mean_magnetisation = np.zeros(shape = (len(seeds), 2, simulations))
mean_abs_mean_magnetisation_std = np.zeros(shape = (len(seeds), 2, simulations))
heat_capacitance = np.zeros(shape = (len(seeds), 2, simulations))
heat_capacitance_std = np.zeros(shape = (len(seeds), 2, simulations)) 

energy_array = np.zeros(shape=(len(seeds), 2, simulations, loops))
mean_magnetisation_array = np.zeros(shape=(len(seeds), 2, simulations, loops))


time_array = np.linspace(0, loops, loops)

for seed in range(0, seeds):
    np.random.seed(seeds[seed])
    for sim in range(0, simulations): # metropolis algorithm
        # Initialization
        spin_lattice = np.random.choice([1, -1], size=(n, n))
        #spin_lattice = np.full((n, n), 1) # all spins are up

        # setting up energy, magnetisation, and time arrays
        energy = calculate_system_energy(spin_lattice)
        energy_array[seed][0][sim][0] = energy
        mean_magnetisation_array[seed][0][sim][0] = np.mean(spin_lattice)

        for loop in tqdm(range(loops), desc=f"Seed {seed+1}/{len(seeds)}Processing Steps Metropolis {sim+1}/{simulations}"): # looping the flips
            spin_lattice = metropolis(sim=sim, loop=loop, seed=seed)

        # getting averages of desired variables
        mean_energy_per_sim[seed][0][sim] = np.mean(energy_array[seed][0][sim])
        mean_energy_per_sim_last[seed][0][sim] = np.mean(energy_array[seed][0][sim][-last:])
        mean_energy_per_sim_last_std[seed][0][sim] = np.sdom(energy_array[seed][0][sim][-last:])
        mean_abs_mean_magnetisation[seed][0][sim] = np.mean(np.abs(mean_magnetisation_array[seed][0][sim][-last:]))
        mean_abs_mean_magnetisation_std[seed][0][sim] = np.std(np.abs(mean_magnetisation_array[seed][0][sim][-last:]))
        heat_capacitance[seed][0][sim] = (mean_energy_per_sim[seed][0][sim])/T[sim]
        heat_capacitance_std[seed][0][sim] = (mean_energy_per_sim_last_std[seed][0][sim])/T[sim]



    np.random.seed(seeds[seed]) # setting the seed again for the same random numbers as the metropolis algorithm
    for sim in range(0, simulations): # wolff algorithm
        # Initialization
        spin_lattice = np.random.choice([1, -1], size=(n, n))
        #spin_lattice = np.full((n, n), 1) # all spins are up

        # setting up energy, magnetisation, and time arrays
        energy = calculate_system_energy(spin_lattice)
        energy_array[seed][1][sim][0] = energy
        mean_magnetisation_array[seed][1][sim][0] = np.mean(spin_lattice)

        for loop in tqdm(range(loops), desc=f"Processing Steps Wolff {sim+1}/{simulations}"):
            spin_lattice = wolff_step(spin_lattice, T, sim, loop, seed)

        # getting averages of desired variables
        mean_energy_per_sim[seed][1][sim] = np.mean(energy_array[seed][1][sim])
        mean_energy_per_sim_last[seed][1][sim] = np.mean(energy_array[seed][1][sim][-last:])
        mean_energy_per_sim_last_std[seed][1][sim] = np.std(energy_array[seed][1][sim][-last:])
        mean_abs_mean_magnetisation[seed][1][sim] = np.mean(np.abs(mean_magnetisation_array[seed][1][sim][-last:]))
        mean_abs_mean_magnetisation_std[seed][1][sim] = np.std(np.abs(mean_magnetisation_array[seed][1][sim][-last:]))
        heat_capacitance[seed][1][sim] = (mean_energy_per_sim_last[seed][1][sim])/T[sim]
        heat_capacitance_std[seed][1][sim] = (mean_energy_per_sim_last_std[seed][1][sim])/T[sim]



# setting up data figures and plots
fig_graph, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(16, 10))
fig_dots, (ax5, ax6, ax7) = plt.subplots(ncols=1, nrows=3, figsize=(16, 10))
for seed in range(0, seeds):
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
    ax2.set_ylabel("Mean magnetisation")
    ax3.set_ylabel("Energy")
    ax4.set_ylabel("Mean magnetisation")
    ax1.set_title("Energy of Metropolis algorithm")
    ax2.set_title("Mean magnetisation of Metropolis algorithm")
    ax3.set_title("Energy of Wolff algorithm")
    ax4.set_title("Mean magnetisation of Wolff algorithm")

    ax5.scatter(T, mean_energy_per_sim_last[seed][0])
    ax5.errorbar(T, mean_energy_per_sim_last[seed][0], yerr=mean_energy_per_sim_last_std[seed][0], fmt='o')
    ax5.scatter(T, mean_energy_per_sim_last[seed][1])
    ax5.errorbar(T, mean_energy_per_sim_last[seed][1], yerr=mean_energy_per_sim_last_std[seed][1], fmt='o')
    ax5.set_xlabel("Effective temperature")
    ax5.set_ylabel("Energy")
    ax6.scatter(T, mean_abs_mean_magnetisation[seed][0])
    ax6.errorbar(T, mean_abs_mean_magnetisation[seed][0], yerr=mean_abs_mean_magnetisation_std[seed][0], fmt='o')
    ax6.scatter(T, mean_abs_mean_magnetisation[seed][1])
    ax6.errorbar(T, mean_abs_mean_magnetisation[seed][1], yerr=mean_abs_mean_magnetisation_std[seed][1], fmt='o')
    ax6.set_xlabel("Effective temperature")
    ax6.set_ylabel("Mean absolute magnetisation")
    ax7.scatter(T, heat_capacitance[seed][0])
    ax7.errorbar(T, heat_capacitance[seed][0], yerr=heat_capacitance_std[seed][0], fmt='o')
    ax7.scatter(T, heat_capacitance[seed][1])
    ax7.errorbar(T, heat_capacitance[seed][1], yerr=heat_capacitance_std[seed][1], fmt='o')
    ax7.set_xlabel("Effective temperature")
    ax7.set_ylabel("Heat capacity")
plt.tight_layout()
fig_graph.savefig("ising_graph_comparison.png")
fig_dots.savefig("ising_data.png")