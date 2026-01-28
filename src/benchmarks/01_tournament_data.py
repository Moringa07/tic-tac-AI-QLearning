import copy
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.ai.ql_agent import GeneticQLAgent
from src.benchmarks.utils import evaluate_vs_minimax
from src.training.gym import train_with_decay

POPULATION_SIZE = 50
GENERATIONS = 30
TOUR_SIZE = 5
MAX_WORKERS = 6
NUM_GAMES = 50
NUM_EPISODES = 5000

MUTATION_START_RATE = 0.30
MUTATION_END_RATE = 0.05

CSV_FILE_GENERATION_DATA = "generations_data.csv"
CSV_FIELDNAMES = ["Generation", "Max_Fitness", "Avg_Fitness", "Optimal_HPs"]
# ----------------------------------------------------


def append_generation_data(data):
    """Añade una fila al archivo CSV de generación."""
    file_exists = os.path.isfile(CSV_FILE_GENERATION_DATA)
    with open(CSV_FILE_GENERATION_DATA, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


def initialize_population():
    return [GeneticQLAgent(i) for i in range(POPULATION_SIZE)]


def evaluate_individual_task(individual_data, generation_num, total_generations, individual_index, total_individuals):
    """
    Función que realiza el entrenamiento y la evaluación de un solo agente.
    """
    individual = individual_data["agent"]
    episodes = individual_data["episodes"]
    num_games = individual_data["num_games"]
    reward_draw_gen = individual.reward_draw
    decay_rate_gen = individual.epsilon_decay_rate
    """
    print(
        f"  [G{generation_num}/{total_generations} - Agente {individual_index + 1:02d}/{total_individuals}] "
        f"Evaluando α={individual.alpha:.2f}, γ={individual.gamma:.2f}..."
    )
    """

    individual.instantiate_agent()
    individual.agent, episodes_to_optimal = train_with_decay(
        individual.agent, episodes=episodes, epsilon_decay_gen=decay_rate_gen, reward_draw_gen=reward_draw_gen
    )

    wins, _, draws = evaluate_vs_minimax(individual.agent, num_games=num_games)
    quality_fitness = (wins + draws) / num_games

    speed_bonus = (episodes - episodes_to_optimal) / episodes

    individual.episodes_to_optimal = episodes_to_optimal

    if quality_fitness < 1.0:
        individual.fitness = quality_fitness
    else:
        individual.fitness = quality_fitness + (speed_bonus * 0.1)

    log_line = (
        f"    -> FITNESS: {individual.fitness:.3f} (W:{wins} D:{draws} L:{num_games - wins - draws}) "
        f"[CONV: {episodes_to_optimal}/{episodes}]"
    )

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

            # print(log_line)


def selection(population):
    best = None
    for _ in range(TOUR_SIZE):
        candidate = random.choice(population)
        if best is None or candidate.fitness > best.fitness:
            best = candidate
    return best


def crossover(parent1, parent2, mutation_rate=0.1):
    p1 = [parent1.alpha, parent1.gamma, parent1.epsilon_decay_rate, parent1.reward_draw]
    p2 = [parent2.alpha, parent2.gamma, parent2.epsilon_decay_rate, parent2.reward_draw]

    crossover_point = random.randint(0, len(p1) - 1)

    child_gen = p1[:crossover_point] + p2[crossover_point:]

    # Mutación
    for i in range(len(child_gen)):
        if random.random() < mutation_rate:
            if i < 3:
                child_gen[i] = round(random.uniform(0.01, 0.99), 2)
            else:
                child_gen[i] = round(random.uniform(0.0, 1.0), 2)

    return GeneticQLAgent(0, gen=child_gen)


def run_genetic_algorithm():
    population = initialize_population()
    best_overall = None
    # all_generation_fitness = [] # No se usa

    if os.path.exists(CSV_FILE_GENERATION_DATA):
        os.remove(CSV_FILE_GENERATION_DATA)

    for generation in range(GENERATIONS):
        evaluate_population(population, generation + 1, GENERATIONS)

        population.sort(key=lambda x: x.fitness, reverse=True)

        current_best = population[0]

        # 1. CÁLCULO DE MÉTRICAS DE LA GENERACIÓN
        current_generation_fitnesses = [p.fitness for p in population]
        avg_fitness = sum(current_generation_fitnesses) / POPULATION_SIZE
        max_fitness = current_best.fitness

        # 2. GUARDAR DATOS DE LA GENERACIÓN
        generation_data = {
            "Generation": generation + 1,
            "Max_Fitness": max_fitness,
            "Avg_Fitness": avg_fitness,
            "Optimal_HPs": f"α={current_best.alpha}, γ={current_best.gamma}, R_draw={current_best.reward_draw}",
        }
        append_generation_data(generation_data)

        # 3. ACTUALIZAR EL MEJOR GLOBAL (Guardando los HPs y la Eficiencia)
        # Acceder a 'fitness' con corchetes
        if best_overall is None or current_best.fitness > best_overall["fitness"]:
            best_overall = {
                "fitness": current_best.fitness,
                "alpha": current_best.alpha,
                "gamma": current_best.gamma,
                "epsilon_decay_rate": current_best.epsilon_decay_rate,
                "reward_draw": current_best.reward_draw,
                # ESTO ES CRUCIAL: GUARDAR EL VALOR ENCONTRADO EN LA EVALUACIÓN
                "episodes_to_optimal": getattr(current_best, "episodes_to_optimal", NUM_EPISODES),
            }

        new_population = []

        NUM_ELITES = 3
        new_population.extend(population[:NUM_ELITES])

        current_mutation_rate = max(
            MUTATION_END_RATE,
            MUTATION_START_RATE - (MUTATION_START_RATE - MUTATION_END_RATE) * (generation / GENERATIONS),
        )

        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)
            child = crossover(parent1, parent2, current_mutation_rate)
            new_population.append(child)

        population = new_population
        for i, agent in enumerate(population):
            agent.id = i

    episodes_to_optimal = best_overall.get("episodes_to_optimal", NUM_EPISODES)

    print("\n--- RESULTADO FINAL ---")
    print(f"Mejor Agente Global: Fitness={best_overall['fitness']:.3f}")
    efficiency_percent = (NUM_EPISODES - episodes_to_optimal) / NUM_EPISODES * 100
    print(f"Eficiencia de Convergencia: {episodes_to_optimal}/{NUM_EPISODES} episodios ({efficiency_percent:.1f}%)")

    print(
        f"Hyperparámetros Óptimos: α={best_overall['alpha']:.2f}, "
        f"γ={best_overall['gamma']:.2f}, "
        f"decay={best_overall['epsilon_decay_rate']:.4f}, "
        f"R_draw={best_overall['reward_draw']:.2f}"
    )


if __name__ == "__main__":
    inicio = time.time()
    run_genetic_algorithm()
    fin = time.time()
    print(fin - inicio)
