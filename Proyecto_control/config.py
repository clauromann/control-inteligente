import os

# ========================================== #
# 1. PARÁMETROS FÍSICOS DEL SISTEMA
# ========================================== #
GRAVITY = 9.81
MASS_CART = 0.768       # M: Masa del carrito (kg)
MASS_POLE = 0.038       # m: Masa del péndulo (kg)
LENGTH = 0.05          # l: Distancia al centro de masas del péndulo (m)
MU_C = 0.5           # Fricción del carrito
MU_P = 0.0227           # Fricción del péndulo

# ========================================== #
# 2. RESTRICCIONES Y CONTROL
# ========================================== #
MAX_VOLTAGE = 0.5     # Límite de los motores (Voltios)
MAX_POSITION = 0.35   # Límite del carril (metros desde el centro)
# Constante imaginaria para convertir Voltios a Fuerza (N). 
# ¡Deberás ajustarla según los motores reales de tu proyecto!
VOLTAGE_TO_FORCE_FACTOR = 9.4 

# ========================================== #
# 3. PARÁMETROS DE SIMULACIÓN Y RL
# ========================================== #
DT = 0.02             # Paso de tiempo de simulación (50Hz)
MAX_EPISODE_STEPS = 750 # Unos 10 segundos de simulación por episodio

# ========================================== #
# 4. RUTAS DE CARPETAS (Directorios)
# ========================================== #
LOG_DIR = "./tensorboard_logs/"
MODEL_DIR = "./saved_models/"
VIDEO_DIR = "./videos/"

# Crear carpetas si no existen
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)