import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from checker import checker # used for testing with a checkerboard pattern as initial system


"================================================================================================================="
# set the seed for the random functions
np.random.seed(42)

animate_sim = False # gives the option to see a realtime animation of the simulation
cooldown = False # activates a function to lower the effective temperature over each iteration

loops = 100000 # amount of spins/chunckspins per simulation
effective_temperature = [4, 5, 6, 7, 8, 9, 10, 15, 25, 50]
simulations = len(effective_temperature)

J = 1 # coupling energy per spin (irrelavent, but used in calculations to show the true formulae, but is canceled at the end)

# Create n by n array of spins
n = 50



"================================================================================================================="

def calculate_system_energy(arr): # used to calculate the energy of the total array put in
    E_sum = 0 # setting the starting energy to 0
    (n, m) = np.shape(arr)

    for i in range(n):
        for j in range(m):
            # checking for neighbors of (i,j)
            k = [i-1, np.mod((i+1), n)]
            l = [j-1, np.mod((j+1), m)]

            # multiplying and adding the energy of the neighbors of (i, j)
            for item_k in k:
                E_sum += -J*arr[i][j] * arr[item_k][j]
            for item_l in l:
                E_sum += -J*arr[i][j] * arr[i][item_l]
    return E_sum

def optimal_calculate_system_energy(arr): # TODO: for wolf algorithm
    pass


def calculate_difference_energy(arr, i, j): # optimized way to calculate the change in energy when a flip is attempted
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

def flip_spin(arr, i, j): # function to invert the value of (i, j) from an array
    arr[i][j] *= -1
    return arr

def try_flip(arr, i, j, T, sim, loop): # attempts to flip an object and flips it if possible
    if cooldown == True: # checks if cooldown is set to true and will reduce the effective temperature over time
        global effective_temperature
        effective_temperature -= effective_temperature/loops
    old_arr = arr.copy() # copies the data from the inserted array to return when a flip fails

    temp_arr, dE = calculate_difference_energy(arr, i, j)
    p = np.exp(-dE/(J*T))
    R = np.random.random()

    global energy_array

    if (dE < 0 or R < p): # flip is benificial for energy or the random chance flips the object
        global energy
        energy += dE
        energy_array[sim][loop] = energy

        return temp_arr
    else: # flipping the object failed
        energy_array[sim][loop] = energy
        return old_arr
    
"================================================================================================================="

# animation code from https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively
def init():
    image_object.set_data(spin_system)
    return image_object

def animate(*iter, sim=1, loop=0):
    # getting random coordinates to flip
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    global spin_system # global is needed to access the variable generated later in the script
    spin_system = try_flip(spin_system, i, j, effective_temperature[sim], sim, loop)

    # calculating mean magnetisation and adding it to the array
    mean_magnetisation = np.mean(spin_system)
    global mean_magnetisation_array # generated later in script
    mean_magnetisation_array[sim][loop] = mean_magnetisation

    if animate_sim == True: # waste of recources when not animating
        image_object.set_data(spin_system)
        return image_object
    else:
        return

"================================================================================================================="

# initializing figure to put in the graphs and final system
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(10, 6))
energy_array = np.zeros(shape=(simulations, loops))
mean_magnetisation_array = np.zeros(shape=(simulations, loops))

# if statement if you want to see a realtime animation of the flip system
if animate_sim == True:
    spin_system = np.full((n, n), 1) # all spins are up
    #spin_system = checker(spin_system) # creates a chekcerboard pattern
    #spin_system = np.full((n, n), -1) # all spins are down
    #spin_system = np.random.choice([1, -1], size=(n, n))

    energy = calculate_system_energy(spin_system) # calculating initial energy of the spin system
    image_object = ax3.imshow((spin_system), cmap='gray', vmin=-1, vmax=1) # putting the final system into axes 3 of the figure
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1, interval=100)
    plt.show()
else:
    for j in range(0, simulations):
        #spin_system = np.full((n, n), 1) # all spins are up
        #spin_system = checker(spin_system) # creates a chekcerboard pattern
        #spin_system = np.full((n, n), -1) # all spins are down
        spin_system = np.random.choice([1, -1], size=(n, n))

        energy = calculate_system_energy(spin_system) # calculating initial energy of the spin system
        energy_array[j][0] = np.array([energy]) # creating the array to save energies over time
        mean_magnetisation_array[j] = np.array([[np.mean(spin_system)]]) # calculating initial magnetisation and creating array for saving future magnetisations

        for i in tqdm(range(loops), desc=f"Simulation {j+1}/{simulations}"): # looping the flips
            animate(sim=j, loop=i)

    time_array = np.linspace(0, loops, loops)

    mean_energy_per_sim = np.zeros(shape = (simulations))
    mean_energy_per_sim_last = np.zeros(shape = (simulations))
    mean_mean_magnetisation = np.zeros(shape = (simulations))
    mean_abs_mean_magnetisation = np.zeros(shape = (simulations))
    mean_energy_square = np.zeros(shape = (simulations))
    heat_capacitance = np.zeros(shape = (simulations))

    for sim in range(simulations):
        mean_energy_per_sim[sim] = np.mean(energy_array[sim])
        mean_energy_per_sim_last[sim] = np.mean(energy_array[sim][-1000:])
        mean_mean_magnetisation[sim] = np.mean(mean_magnetisation_array[sim])
        mean_abs_mean_magnetisation[sim] = np.mean(np.abs(mean_magnetisation_array[sim]))
        mean_energy_square[sim] = np.mean(np.square(energy_array[sim]))
        heat_capacitance[sim] = (mean_energy_square[sim] - np.square(mean_energy_per_sim[sim]))/effective_temperature[sim]

        ax1.plot(time_array, energy_array[sim])
        ax2.plot(time_array, mean_magnetisation_array[sim])

        print(f'T = {effective_temperature[sim]}; <E> = {mean_energy_per_sim[sim]}; sd = {np.std(energy_array[sim])}; <m> = {mean_mean_magnetisation[sim]}; sd = {np.std(mean_magnetisation_array[sim])}; <|m|> = {mean_abs_mean_magnetisation[sim]}; sd = {np.std(np.abs(mean_magnetisation_array[sim]))}; C = {heat_capacitance[sim]}')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Magnetisation")

    fig.tight_layout()
    plt.savefig("simulation_graph.png")
    plt.show()


# g.niforos@uu.nl