import numpy as np
from PIL import Image, ImageDraw


IMAGE_SIZE = 75
POPULATION_SIZE = 50
GENERATIONS = 100

CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.005

NUM_DESCENDANTS = 20


def create_pattern():

    image = Image.new(
        "1",
        (IMAGE_SIZE, IMAGE_SIZE),
        1
    )

    draw = ImageDraw.Draw(image)

    # Draw triangle
    draw.polygon(
        [
            (37, 10),
            (15, 60),
            (60, 60)
        ],
        outline=0,
        width=3
    )

    return image



def transform_pattern(pattern):

    # Random scaling
    scale = np.random.uniform(0.6, 1.2)

    new_size = max(
        1,
        int(IMAGE_SIZE * scale)
    )

    scaled = pattern.resize(
        (new_size, new_size),
        Image.Resampling.NEAREST
    )

    # Random rotation
    angle = np.random.uniform(
        -45,
        45
    )

    rotated = scaled.rotate(
        angle,
        expand=True,
        fillcolor=1
    )

    # Create blank image
    canvas = Image.new(
        "1",
        (IMAGE_SIZE, IMAGE_SIZE),
        1
    )

    # Calculate possible translation
    max_x = max(
        0,
        IMAGE_SIZE - rotated.width
    )

    max_y = max(
        0,
        IMAGE_SIZE - rotated.height
    )

    x = np.random.randint(
        0,
        max_x + 1
    )

    y = np.random.randint(
        0,
        max_y + 1
    )

    canvas.paste(
        rotated,
        (x, y)
    )

    return canvas


# ============================================================
# 3. IMAGE TO BINARY CHROMOSOME
# 75 x 75 = 5625 genes
# ============================================================

def image_to_chromosome(image):

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    array = np.array(
        image,
        dtype=np.uint8
    )

    return array.flatten()


# ============================================================
# 4. CREATE PATTERN DESCENDANTS
# ============================================================

def create_descendants(pattern):

    descendants = []

    for _ in range(NUM_DESCENDANTS):

        transformed = transform_pattern(
            pattern
        )

        chromosome = image_to_chromosome(
            transformed
        )

        descendants.append(
            chromosome
        )

    return np.array(
        descendants
    )


# ============================================================
# 5. HAMMING DISTANCE
# ============================================================

def hamming_distance(a, b):

    return np.mean(
        a != b
    )


# ============================================================
# 6. FITNESS FUNCTION
# ============================================================

def calculate_fitness(
    chromosome,
    descendants,
    master
):

    # Distance from descendants
    descendant_distance = np.mean(
        [
            hamming_distance(
                chromosome,
                d
            )
            for d in descendants
        ]
    )

    # Distance from master image
    master_distance = hamming_distance(
        chromosome,
        master
    )

    # Average distance
    fitness = (
        descendant_distance
        +
        master_distance
    ) / 2

    return fitness


# ============================================================
# 7. INITIALIZE POPULATION
# ============================================================

def initialize_population():

    chromosome_length = (
        IMAGE_SIZE * IMAGE_SIZE
    )

    population = np.random.randint(
        0,
        2,
        size=(
            POPULATION_SIZE,
            chromosome_length
        ),
        dtype=np.uint8
    )

    return population


# ============================================================
# 8. ANNULAR CROSSOVER
# ============================================================

def annular_crossover(
    parent1,
    parent2
):

    # No crossover
    if np.random.random() > CROSSOVER_RATE:

        return (
            parent1.copy(),
            parent2.copy()
        )

    length = len(parent1)

    # Half chromosome
    segment_length = length // 2

    # Random starting point
    start = np.random.randint(
        0,
        length
    )

    # Circular indices
    indices = (
        np.arange(
            start,
            start + segment_length
        )
        % length
    )

    child1 = parent1.copy()
    child2 = parent2.copy()

    # Exchange genetic material
    child1[indices] = parent2[indices]
    child2[indices] = parent1[indices]

    return child1, child2


# ============================================================
# 9. MUTATION
# ============================================================

def mutation(chromosome):

    mutation_mask = (
        np.random.random(
            len(chromosome)
        )
        < MUTATION_RATE
    )

    # Flip binary values
    chromosome[mutation_mask] = (
        1 -
        chromosome[mutation_mask]
    )

    return chromosome


# ============================================================
# 10. SELECTION
# Best with Worst
# Second Best with Second Worst
# ============================================================

def create_offspring(
    population,
    fitness_values
):

    # Sort by fitness
    sorted_indices = np.argsort(
        fitness_values
    )

    sorted_population = (
        population[
            sorted_indices
        ]
    )

    offspring = []

    left = 0
    right = POPULATION_SIZE - 1

    while len(offspring) < POPULATION_SIZE:

        parent1 = (
            sorted_population[left]
        )

        parent2 = (
            sorted_population[right]
        )

        # Crossover
        child1, child2 = (
            annular_crossover(
                parent1,
                parent2
            )
        )

        # Mutation
        child1 = mutation(child1)
        child2 = mutation(child2)

        offspring.append(child1)
        offspring.append(child2)

        left += 1
        right -= 1

    return np.array(
        offspring[:POPULATION_SIZE]
    )


# ============================================================
# 11. ELITISM
# ============================================================

def apply_elitism(
    population,
    offspring,
    descendants,
    master
):

    # Combine old and new population
    combined = np.vstack(
        (
            population,
            offspring
        )
    )

    # Calculate fitness
    fitness_values = np.array(
        [
            calculate_fitness(
                chromosome,
                descendants,
                master
            )
            for chromosome in combined
        ]
    )

    # Select best 50
    best_indices = np.argsort(
        fitness_values
    )[:POPULATION_SIZE]

    return combined[
        best_indices
    ]


# ============================================================
# 12. GENETIC ALGORITHM
# ============================================================

def genetic_algorithm(
    master,
    descendants
):

    # Initial population
    population = (
        initialize_population()
    )

    best_score = float("inf")
    best_chromosome = None

    print("\nStarting Genetic Algorithm...\n")

    # Evolution
    for generation in range(
        GENERATIONS
    ):

        # Calculate fitness
        fitness_values = np.array(
            [
                calculate_fitness(
                    chromosome,
                    descendants,
                    master
                )
                for chromosome in population
            ]
        )

        # Find best chromosome
        best_index = np.argmin(
            fitness_values
        )

        current_best = (
            fitness_values[
                best_index
            ]
        )

        # Store global best
        if current_best < best_score:

            best_score = current_best

            best_chromosome = (
                population[
                    best_index
                ].copy()
            )

        print(
            f"Generation "
            f"{generation + 1:3d} | "
            f"Best Distance: "
            f"{best_score:.5f}"
        )

        # Selection + crossover + mutation
        offspring = create_offspring(
            population,
            fitness_values
        )

        # Elitism
        population = apply_elitism(
            population,
            offspring,
            descendants,
            master
        )

    return (
        best_chromosome,
        best_score
    )


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Create original pattern
    # --------------------------------------------------------

    pattern = create_pattern()

    pattern.save(
        "original_pattern.png"
    )

    print(
        "Original pattern created."
    )

    # --------------------------------------------------------
    # Create transformed patterns
    # --------------------------------------------------------

    descendants = create_descendants(
        pattern
    )

    print(
        "Pattern descendants created:",
        len(descendants)
    )

    # --------------------------------------------------------
    # Create master image
    # --------------------------------------------------------

    master_image = transform_pattern(
        pattern
    )

    master_image.save(
        "master_image.png"
    )

    master = image_to_chromosome(
        master_image
    )

    print(
        "Master image created."
    )

    # --------------------------------------------------------
    # Run Genetic Algorithm
    # --------------------------------------------------------

    best_chromosome, best_score = (
        genetic_algorithm(
            master,
            descendants
        )
    )

    # --------------------------------------------------------
    # Display final result
    # --------------------------------------------------------

    print("\n================================")
    print("GENETIC ALGORITHM COMPLETED")
    print("================================")

    print(
        "Final Best Distance:",
        round(best_score, 5)
    )

    # --------------------------------------------------------
    # Convert chromosome back to image
    # --------------------------------------------------------

    result_array = (
        best_chromosome.reshape(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
        * 255
    ).astype(
        np.uint8
    )

    result_image = Image.fromarray(
        result_array
    )

    result_image.save(
        "ga_result.png"
    )

    print("\nOutput files:")
    print("1. original_pattern.png")
    print("2. master_image.png")
    print("3. ga_result.png")

    print(
        "\nPattern recognition completed successfully!"
    )
