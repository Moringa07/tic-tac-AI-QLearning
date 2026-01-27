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

CSV_FILE = "queries/resultados_torneo_final.csv"
FIELDNAMES = ["Run", "Max_Fitness", "Alpha", "Gamma", "Epsilon_Decay", "R_Draw", "Episodes_to_Optimal"]


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


GLOBAL_RUN_COUNTER = 1


def run_genetic_algorithm():
    global GLOBAL_RUN_COUNTER
    population = initialize_population()
    best_overall = None

    for generation in range(GENERATIONS):  # GENERATIONS es el total
        # print(f"\n--- GENERACIÓN {generation + 1}/{GENERATIONS} ---")

        evaluate_population(population, generation + 1, GENERATIONS)

        population.sort(key=lambda x: x.fitness, reverse=True)

        current_best = population[0]
        if best_overall is None or current_best.fitness > best_overall.fitness:
            best_overall = current_best

        new_population = []

        NUM_ELITES = 3

        new_population.extend(population[:NUM_ELITES])

        current_mutation_rate = max(
            MUTATION_END_RATE,
            MUTATION_START_RATE - (MUTATION_START_RATE - MUTATION_END_RATE) * (generation / GENERATIONS),
        )

        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(
                population,
            )
            parent2 = selection(population)
            child = crossover(parent1, parent2, current_mutation_rate)
            new_population.append(child)

        population = new_population
        for i, agent in enumerate(population):
            agent.id = i

    print("\n--- RESULTADO FINAL ---")
    print(f"Mejor Agente Global: Fitness={best_overall.fitness:.3f}")

    print(
        f"Hyperparámetros Óptimos: α={best_overall.alpha:.2f}, "
        f"γ={best_overall.gamma:.2f}, "
        f"decay={best_overall.epsilon_decay_rate}, "
        f"R_draw={best_overall.reward_draw:.2f}"
    )

    results = {
        "Run": GLOBAL_RUN_COUNTER,
        "Max_Fitness": best_overall.fitness,
        "Alpha": best_overall.alpha,
        "Gamma": best_overall.gamma,
        "Epsilon_Decay": best_overall.epsilon_decay_rate,
        "R_Draw": best_overall.reward_draw,
        "Episodes_to_Optimal": getattr(best_overall, "episodes_to_optimal", "N/A"),
    }

    append_to_csv(results)

    GLOBAL_RUN_COUNTER += 1


def append_to_csv(data):
    """Añade una fila al archivo CSV, creando la cabecera si no existe."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)


if __name__ == "__main__":
    inicio = time.time()
    NUM_TOTAL_RUNS = 50
    for i in range(1, NUM_TOTAL_RUNS + 1):
        print(f"\n======== INICIANDO EJECUCIÓN {i}/{NUM_TOTAL_RUNS} ========")
        run_genetic_algorithm()
    print("\nFIN DE LAS EJECUCIONES. Resultados guardados en resultados_torneo_final.csv")
    fin = time.time()
    print(fin - inicio)
