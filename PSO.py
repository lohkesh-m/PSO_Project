import numpy as np

def objective_func(x):

    y=(x[0]-1)**2+(x[1]-1)**2-x[0]*x[1]
    #y=(x[0]+2*x[1]-7)**2+(2*x[0]+x[1]-5)**2  
    # y =  x[0]+(2-x[1])**2
    return y

def PSO(objective_func, num_particles, num_dimensions):
    # Set the parameters for the PSO algorithm
    max_iter = 1000  # maximum number of iterations
    c1 = 2.0  # cognitive parameter
    c2 = 2.0  # social parameter
    w = 0.7  # inertia weight
    search_space = [-5.0, 5.0]  # the search space limits for the particles

    # Initialize the particle positions and velocities
    particles_pos = np.random.uniform(search_space[0], search_space[1], size=(num_particles, num_dimensions))
    particles_vel = np.zeros((num_particles, num_dimensions))

    # Initialize the personal best positions for each particle
    particles_pbest = particles_pos.copy()

    # Initialize the global best position and the fitness value
    gbest_pos = np.zeros(num_dimensions)
    gbest_val = np.inf

    # Run the PSO algorithm
    for i in range(max_iter):
        # Evaluate the fitness values for all particles
        fitness_vals = np.apply_along_axis(objective_func, 1, particles_pos)

        # Update the personal best positions for each particle
        for j in range(num_particles):
            if fitness_vals[j] < objective_func(particles_pbest[j]):
                particles_pbest[j] = particles_pos[j].copy()

        # Update the global best position and the fitness value
        min_fitness_idx = np.argmin(fitness_vals)
        if fitness_vals[min_fitness_idx] < gbest_val:
            gbest_val = fitness_vals[min_fitness_idx]
            gbest_pos = particles_pos[min_fitness_idx].copy()

        # Update the velocities and positions for all particles
        r1 = np.random.rand(num_particles, num_dimensions)
        r2 = np.random.rand(num_particles, num_dimensions)
        particles_vel = w * particles_vel + \
                        c1 * r1 * (particles_pbest - particles_pos) + \
                        c2 * r2 * (gbest_pos - particles_pos)
        particles_pos = particles_pos + particles_vel

        # Enforce the search space limits for the particles
        particles_pos = np.clip(particles_pos, search_space[0], search_space[1])

    # Return the global best position and the fitness value
    return gbest_pos, gbest_val

# Call the PSO function to optimize the objective function
gbest_pos, gbest_val = PSO(objective_func, num_particles=50, num_dimensions=2)

# Print the results
print("Global best position:", gbest_pos)
print("Global best value:", gbest_val)
