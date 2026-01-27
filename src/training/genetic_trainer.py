import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.ai.ql_agent import GeneticQLAgent
from src.benchmarks.utils import evaluate_vs_minimax
from src.training.gym import train_with_decay

POPULATION_SIZE = 30
GENERATIONS = 10
TOUR_SIZE = 5
MAX_WORKERS = 6
NUM_GAMES = 50
NUM_EPISODES = 5000


def initialize_population():
    return [GeneticQLAgent(i) for i in range(POPULATION_SIZE)]


def evaluate_individual_task(individual_data, generation_num, total_generations, individual_index, total_individuals):
    """
    Función que realiza el entrenamiento y la evaluación de UN solo agente.
    Se ejecuta en un proceso separado.
    """
    individual = individual_data["agent"]
    episodes = individual_data["episodes"]
    num_games = individual_data["num_games"]

    print(
        f"  [G{generation_num}/{total_generations} - Agente {individual_index + 1:02d}/{total_individuals}] "
        f"Evaluando α={individual.alpha:.2f}, γ={individual.gamma:.2f}..."
    )

    individual.instantiate_agent()
    individual.agent = train_with_decay(individual.agent, episodes=episodes)

    wins, _, draws = evaluate_vs_minimax(individual.agent, num_games=num_games)

    fitness = (wins + draws) / num_games
    log_line = f"    -> FITNESS: {fitness:.3f} (W:{wins} D:{draws} L:{num_games - wins - draws})"

    individual.fitness = fitness
    return individual, log_line


def evaluate_population(population, generation_num, total_generations):
    total_individuals = len(population)

    tasks = []
    for i, individual in enumerate(population):
        task_data = {
            "agent": individual,
            "episodes": NUM_EPISODES,
            "num_games": NUM_GAMES,
        }
        tasks.append((task_data, generation_num, total_generations, i, total_individuals))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_agent = {executor.submit(evaluate_individual_task, *task): i for i, task in enumerate(tasks)}

        for future in as_completed(future_to_agent):
            updated_individual, log_line = future.result()
            original_index = future_to_agent[future]
            population[original_index] = updated_individual

            print(log_line)


def selection(population):
    best = None
    for _ in range(TOUR_SIZE):
        candidate = random.choice(population)
        if best is None or candidate.fitness > best.fitness:
            best = candidate
    return best


def crossover(parent1, parent2):
    # Crossover de punto único: Mezclamos los genes (hyperparámetros)
    p1 = [parent1.alpha, parent1.gamma, parent1.epsilon_decay_rate]
    p2 = [parent2.alpha, parent2.gamma, parent2.epsilon_decay_rate]

    crossover_point = random.randint(0, len(p1) - 1)

    child_gen = p1[:crossover_point] + p2[crossover_point:]

    for i in range(len(child_gen)):
        if random.random() < 0.1:
            child_gen[i] = round(random.uniform(0.01, 0.99), 2)

    return GeneticQLAgent(0, gen=child_gen)


def run_genetic_algorithm():
    population = initialize_population()
    best_overall = None

    for generation in range(GENERATIONS):  # GENERATIONS es el total
        print(f"\n--- GENERACIÓN {generation + 1}/{GENERATIONS} ---")

        evaluate_population(population, generation + 1, GENERATIONS)

        current_best = max(population, key=lambda x: x.fitness)
        print(f"Mejor Fitness: {current_best.fitness:.3f} (α={current_best.alpha}, γ={current_best.gamma})")

        if best_overall is None or current_best.fitness > best_overall.fitness:
            best_overall = current_best

        new_population = []

        new_population.append(current_best)

        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2)
            new_population.append(child)

        population = new_population
        # Actualizar IDs
        for i, agent in enumerate(population):
            agent.id = i

    print("\n--- RESULTADO FINAL ---")
    print(f"Mejor Agente Global: Fitness={best_overall.fitness:.3f}")
    print(
        f"Hyperparámetros Óptimos: α={best_overall.alpha}, γ={best_overall.gamma}, decay={best_overall.epsilon_decay_rate}"
    )


if __name__ == "__main__":
    inicio = time.time()
    run_genetic_algorithm()
    fin = time.time()
    print(fin - inicio)

"""
Mejor Agente Global: Fitness=1.000
Hyperparámetros Óptimos: α=0.28, γ=0.48, decay=0.0051
1644.6024887561798
Mejor Agente Global: Fitness=1.000
Hyperparámetros Óptimos: α=0.29, γ=0.04, decay=0.0003
1786.7330901622772
"""
