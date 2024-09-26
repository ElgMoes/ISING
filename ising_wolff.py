import numpy as np
from icecream import ic  # debugging tool
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

# Constants and setup
np.random.seed(42)
animate_sim = False  # Show live animation if True
save_animation = False # very slow performance (40 times slower), this will not work when live animation is turned on
n = 50  # Size of the lattice (n x n)
loops = 100000  # Number of flipping attempts
T = [4, 5, 6, 7, 8, 9, 10, 15, 25, 50]
simulations = len(T)

def calculate_system_energy(arr):
    # Nearest neighbor shifts (using periodic boundary conditions with np.roll)
    right = np.roll(arr, -1, axis=1)
    left = np.roll(arr, 1, axis=1)
    up = np.roll(arr, -1, axis=0)
    down = np.roll(arr, 1, axis=0)

    # Calculate energy by summing interactions with neighbours
    E_sum = -np.sum(arr * (right + left + up + down))
    return E_sum

# main flipping function, creates and flips clusers
def flip_wolff_cluster(lattice, i, j, T, sim=1, loop=1):
    old_lattice = lattice.copy() # saving it for when flip fails
    initial_spin = lattice[i, j]
    lattice[i, j] *= -1 # flips first object
    p_add = 1 - np.exp(-2 / T) # generates possibility of neighbouring object to be added to cluster
    stack = [(i, j)] # stack of cluster coordinates
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    (n, m) = np.shape(lattice)

    # Create a mask for visited lattice sites to avoid duplicates
    visited = np.zeros_like(lattice, dtype=bool)
    visited[i, j] = True  # Mark the first flipped site as visited

    # loops over stack to check for new possible cluster members
    while stack:
        (x, y) = stack.pop()
        for dx, dy in neighbours:
            nx, ny = (x + dx) % n, (y + dy) % m # creating new coordinates to be added to the cluster
            if not visited[nx, ny] and lattice[nx, ny] == initial_spin and np.random.random() < p_add: # checks if new coordinate has the same spin and is below the probability threshold
                lattice[nx, ny] *= -1 # flips the newly added object
                visited[nx, ny] = True # States that the location has been visited
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
        energy_array[sim][loop] = energy
        return lattice # returns new lattice with flipped cluster
    else:
        energy_array[sim][loop] = energy
        return old_lattice # returns old lattice when cluster flip failed

# loops over the cluster creation and flipping
def wolff_step(lattice, T, sim=1, loop=1):
    T = T[sim]
    i, j = np.random.randint(0, n, size=2) # generates random coordinates
    lattice = flip_wolff_cluster(lattice, i, j, T, sim, loop)

    # storing magnetisation over every step
    global mean_magnetisation_array
    mean_magnetisation_array[sim][loop] = np.mean(lattice)
    return lattice

# init function for the animation
def init():
    image_object.set_data(spin_lattice)
    return image_object

def animate(frame): # this should be obvious
    global spin_lattice
    spin_lattice = wolff_step(spin_lattice, T)
    image_object.set_data(spin_lattice)
    pbar.update(1)  # Update tqdm progress bar
    return image_object

# setting up arrays to store desired variables
mean_energy_per_sim = np.zeros(shape = (simulations))
mean_energy_per_sim_last = np.zeros(shape = (simulations))
mean_mean_magnetisation = np.zeros(shape = (simulations))
mean_abs_mean_magnetisation = np.zeros(shape = (simulations))
mean_energy_square = np.zeros(shape = (simulations))
heat_capacitance = np.zeros(shape = (simulations))

if animate_sim: # animates the simulation in real time
    fig, ax = plt.subplots()
    image_object = ax.imshow(spin_lattice, cmap='gray', vmin=-1, vmax=1)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=loops, interval=50)
    plt.show()
else:
        if save_animation: # stores the animation in a video file
            # Create a tqdm progress bar
            with tqdm(total=loops, desc="Creating Animation") as pbar:
                fig, ax = plt.subplots()
                image_object = ax.imshow(spin_lattice, cmap='gray', vmin=-1, vmax=1)
                # Update `animate` function to use the progress bar
                anim = animation.FuncAnimation(fig, animate, init_func=init, frames=loops, interval=50)
                anim.save('simulation_animation.mp4', writer='ffmpeg')  # Save with filename
                plt.close() # needed to close the animation and keep it from doing live updates
        else: # just loops over the steps
            energy_array = np.zeros(shape=(simulations, loops))
            mean_magnetisation_array = np.zeros(shape=(simulations, loops))
            for sim in range(0, simulations):
                # Initialization
                #spin_lattice = np.random.choice([1, -1], size=(n, n))
                spin_lattice = np.full((n, n), 1) # all spins are up
                very_old_lattice = spin_lattice.copy()
                # setting up energy, magnetisation, and time arrays
                energy = calculate_system_energy(spin_lattice)
                energy_array[sim][0] = energy
                mean_magnetisation_array[sim][0] = np.mean(spin_lattice)
                time_array = np.linspace(0, loops, loops)

                # getting averages of desired variables
                mean_energy_per_sim[sim] = np.mean(energy_array[sim])
                mean_energy_per_sim_last[sim] = np.mean(energy_array[sim][-1000:])
                mean_mean_magnetisation[sim] = np.mean(mean_magnetisation_array[sim])
                mean_abs_mean_magnetisation[sim] = np.mean(np.abs(mean_magnetisation_array[sim]))
                mean_energy_square[sim] = np.mean(np.square(energy_array[sim]))
                heat_capacitance[sim] = (mean_energy_square[sim] - np.square(mean_energy_per_sim[sim]))/T[sim]

                for loop in tqdm(range(loops), desc=f"Processing Steps {sim+1}/{simulations}"):
                    spin_lattice = wolff_step(spin_lattice, T, sim, loop)

                print(f'T = {T[sim]}; <E> = {mean_energy_per_sim[sim]}; sd = {np.std(energy_array[sim])}; <m> = {mean_mean_magnetisation[sim]}; sd = {np.std(mean_magnetisation_array[sim])}; <|m|> = {mean_abs_mean_magnetisation[sim]}; sd = {np.std(np.abs(mean_magnetisation_array[sim]))}; C = {heat_capacitance[sim]}')

        ic(energy_array)
        # making a nice plot of the data
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
        for sim in range(simulations):
            ax1.plot(time_array, energy_array[sim])
            ax2.plot(time_array, mean_magnetisation_array[sim])
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Energy")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Magnetization")
        ax3.imshow(very_old_lattice, cmap='gray', vmin=-1, vmax=1)
        ax3.set_title(f"Old Lattice {simulations}")
        ax4.imshow(spin_lattice, cmap='gray', vmin=-1, vmax=1)
        ax4.set_title(f"New Lattice {simulations}")
        ax1.savefig("energy_plot.png")
        ax2.savefig("mean_magnetism_plot.png")
        plt.tight_layout()
        plt.show()
