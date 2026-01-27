import os

from src.ai.gym import train


def generate_models():
    milestones = [100, 200, 500, 1000, 1500, 2500, 5000, 10000, 20000]
    model_dir = "src/models"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("Iniciando entrenamiento evolutivo...")

    for episodes in milestones:
        print(f"Entrenando modelo de {episodes} episodios...")
        agent = train(episodes=episodes)

        filename = os.path.join(model_dir, f"q_table_{episodes}.pkl")
        agent.save_model(filename)
        print(f"Modelo guardado en: {filename}")


if __name__ == "__main__":
    generate_models()
