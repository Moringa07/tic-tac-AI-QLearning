CONTAINER_NAME = tictactoe_dev
.PHONY: help build start stop run simulate notebook install shell lint lint-fix lint-unsafe format clean-code prune pre-commit-install

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Makefile de Ayuda para el Proyecto Tres en Raya"
	@echo "------------------------------------------------"
	@echo "Uso: make [target]"
	@echo ""
	@echo "🎯 Targets Principales:"
	@printf "  \033[36m%-18s\033[0m %s\n" "build" "Construye la imagen y levanta el contenedor por primera vez."
	@printf "  \033[36m%-18s\033[0m %s\n" "start" "Inicia el contenedor si está detenido."
	@printf "  \033[36m%-18s\033[0m %s\n" "stop" "Detiene y elimina el contenedor."
	@printf "  \033[36m%-18s\033[0m %s\n" "run" "Jugar: Ejecuta la interfaz gráfica (src/main.py)."
	@printf "  \033[36m%-18s\033[0m %s\n" "benchmark" "Estadísticas: Ejecuta las simulaciones (src/benchmark.py)."
	@printf "  \033[36m%-18s\033[0m %s\n" "notebook" "Análisis: Lanza Jupyter Lab en el navegador."
	@echo ""
	@echo "💎 Calidad de Código (Ruff):"
	@printf "  \033[36m%-18s\033[0m %s\n" "lint" "Solo revisa errores (sin modificar nada)."
	@printf "  \033[36m%-18s\033[0m %s\n" "lint-fix" "Revisa y ARREGLA errores seguros (imports, estilo)."
	@printf "  \033[36m%-18s\033[0m %s\n" "lint-unsafe" "LIMPIEZA PROFUNDA: Borra variables no usadas (agresivo)."
	@printf "  \033[36m%-18s\033[0m %s\n" "format" "Re-ordena el código para que se vea bonito."
	@printf "  \033[36m%-18s\033[0m %s\n" "clean-code" "Combo: Ejecuta format + lint-fix."
	@echo ""
	@echo "🛠️ Desarrollo y Configuración:"
	@printf "  \033[36m%-18s\033[0m %s\n" "install" "Sincroniza dependencias (uv sync)."
	@printf "  \033[36m%-18s\033[0m %s\n" "shell" "Entra a la terminal del contenedor."
	@printf "  \033[36m%-18s\033[0m %s\n" "pre-commit-install" "Instala los git hooks para chequeo automático."
	@printf "  \033[36m%-18s\033[0m %s\n" "prune" "¡PELIGRO! Borra todas las imágenes y contenedores de Podman."
	@echo ""

# --- Construcción y Ejecución ---

build: ## Construye y levanta el contenedor desde cero
	@echo "-> Construyendo y levantando el entorno de desarrollo..."
	@podman-compose up -d --build
	@make install

start: ## Inicia el contenedor si ya existe
	@echo "-> Iniciando el contenedor..."
	@podman-compose up -d

stop: ## Detiene y elimina el contenedor
	@echo "-> Deteniendo el contenedor..."
	@podman-compose down

run: ## Ejecuta el script principal
	@echo "-> Concediendo acceso a la pantalla (X11)..."
	@xhost +local:
	@echo "-> Ejecutando el juego..."
	@podman exec -it $(CONTAINER_NAME) uv run src/main.py

benchmark: ## Ejecuta una simulación de N partidas para recolectar datos
	@echo "-> Iniciando la simulación estadística (esto puede tardar)..."
	@podman exec -it $(CONTAINER_NAME) uv run src/benchmark.py

notebook: ## Lanza un servidor de Jupyter Lab para el análisis de datos
	@echo "-> Lanzando servidor de Jupyter Lab..."
	@echo "-> Copia la URL que aparecerá a continuación en tu navegador."
	@podman exec -it $(CONTAINER_NAME) jupyter lab \
		--ip=0.0.0.0 \
		--port=8888 \
		--no-browser \
		--allow-root

# --- Gestión de Dependencias y Entorno ---

install: ## Sincroniza las dependencias (main Y dev)
	@echo "-> Sincronizando dependencias con uv (incluyendo --dev)..."
	@podman exec -it $(CONTAINER_NAME) uv sync --dev

shell: ## Entra al contenedor
	@echo "-> Abriendo terminal en el contenedor '$(CONTAINER_NAME)'..."
	@podman exec -it $(CONTAINER_NAME) /bin/bash

pre-commit-install: ## Instala los git hooks en tu repositorio local
	@echo "-> Instalando git hooks con pre-commit..."
	@podman exec -it $(CONTAINER_NAME) pre-commit install

prune: ## Limpieza profunda de Podman
	@echo "-> Limpiando el sistema Podman (imágenes, contenedores, volúmenes)..."
	@podman system prune -a --volumes --force

# --- Calidad de Código y Limpieza (Ruff) ---

lint: ## Solo revisa, no toca nada
	@echo "-> Revisando el código con Ruff (Solo lectura)..."
	@podman exec -it $(CONTAINER_NAME) ruff check .

lint-fix: ## Arregla cosas seguras (imports, espacios)
	@echo "-> Aplicando correcciones seguras con Ruff..."
	@podman exec -it $(CONTAINER_NAME) ruff check . --fix

lint-unsafe: ## Arregla variables no usadas (Agresivo)
	@echo "-> Aplicando correcciones AGRESIVAS (variables no usadas)..."
	@podman exec -it $(CONTAINER_NAME) ruff check . --fix --unsafe-fixes

format: ## Formatea el código
	@echo "-> Formateando el código con Ruff..."
	@podman exec -it $(CONTAINER_NAME) ruff format .

clean-code: ## Combo: Formatear y Arreglar
	@echo "-> Limpiando y embelleciendo todo el código..."
	@make format
	@make lint-fix
