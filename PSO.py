
import numpy as np

# PSO Scheduling Application

num_tasks = 5
num_slots = 5
num_particles = 20
iterations = 50

w = 0.5
c1 = 1.5
c2 = 1.5

# Fitness function to count scheduling conflicts
def fitness(position):
    schedule = np.round(position).astype(int)
    schedule = np.clip(schedule, 0, num_slots - 1)

    conflicts = 0

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            if schedule[i] == schedule[j]:
                conflicts += 1

    return conflicts

# Initialize particles
positions = np.random.uniform(
    0, num_slots - 1, (num_particles, num_tasks)
)

velocities = np.random.uniform(
    -1, 1, (num_particles, num_tasks)
)

pbest = positions.copy()
pbest_fitness = np.array([fitness(p) for p in positions])

gbest_index = np.argmin(pbest_fitness)
gbest = pbest[gbest_index].copy()

# Particle Swarm Optimization
for iteration in range(iterations):

    for i in range(num_particles):

        r1 = np.random.rand(num_tasks)
        r2 = np.random.rand(num_tasks)

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest[i] - positions[i])
            + c2 * r2 * (gbest - positions[i])
        )

        positions[i] += velocities[i]
        positions[i] = np.clip(
            positions[i], 0, num_slots - 1
        )

        current_fitness = fitness(positions[i])

        if current_fitness < pbest_fitness[i]:
            pbest[i] = positions[i].copy()
            pbest_fitness[i] = current_fitness

    gbest_index = np.argmin(pbest_fitness)
    gbest = pbest[gbest_index].copy()

    print("Iteration", iteration + 1,
          "Best Cost:", fitness(gbest))

# Display final schedule
best_schedule = np.round(gbest).astype(int)

print("\n----- OPTIMIZED SCHEDULE -----")

for i in range(num_tasks):
    print("Task", i + 1,
          "-> Time Slot", best_schedule[i] + 1)

print("\nMinimum Scheduling Cost:", fitness(gbest))

if fitness(gbest) == 0:
    print("Scheduling Completed Successfully!")
else:
    print("Scheduling Conflicts Remain.")
