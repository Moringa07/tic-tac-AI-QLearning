WIDTH, HEIGHT = 1100, 750  # Expandido para márgenes cómodos
BOARD_WIDTH = 600
BOARD_OFFSET_Y = 100  # Margen superior
BOARD_OFFSET_X = 50  # Margen izquierdo
BOARD_ROWS, BOARD_COLS = 3, 3
SQUARE_SIZE = BOARD_WIDTH // BOARD_COLS
LINE_WIDTH = 15


# Configuración de Colores - Medium Celeste Theme
BG_COLOR = (50, 80, 100)      # Medium Celeste/Slate (Steel Blue)
BOARD_BG_COLOR = (40, 60, 80) 
LINE_COLOR = (255, 165, 0)    # Bright Orange (Keep accent)
SQUARE_HOVER_COLOR = (70, 100, 120) 

CIRCLE_COLOR = (0, 255, 230)   # Cyan Vibrante (O)
CROSS_COLOR = (255, 120, 170)  # Pink/Magenta (X)
WIN_LINE_COLOR = (255, 215, 0) # Gold

# Dimensiones de Botones para Menú Simétrico
BUTTON_WIDTH = 400
BUTTON_HEIGHT = 60
BUTTON_MARGIN = 20

# Tamaño de tableros para modo pantalla dividida
SPLIT_BOARD_WIDTH = 400

CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25

FONT_SIZE = 50
FONT_COLOR = (230, 230, 230) # Texto claro suave
MENU_FONT_SIZE = 50          # Ajustado para que quepan mejor
MENU_SELECTED_COLOR = (255, 193, 7) # Seleccionado (Gold)
MENU_BUTTON_COLOR = (55, 60, 80)    # Color base de botones
MENU_BUTTON_HOVER = (75, 80, 100)   # Color hover de botones

FPS = 60

# Constantes del Grafo
GRAPH_BG_COLOR = (30, 33, 48)
NODE_COLOR = (200, 200, 200)
NODE_RADIUS = 20
TEXT_COLOR = (255, 255, 255)
POSITIVE_COLOR = (0, 230, 118)
NEGATIVE_COLOR = (255, 64, 129)
NEUTRAL_COLOR = (120, 120, 140)


class PlayerType:
    HUMAN = "HUMAN"
    AI_SLOW = "AI_SLOW"
    AI_FAST = "AI_FAST"
    AI_QL = "AI_QL"
