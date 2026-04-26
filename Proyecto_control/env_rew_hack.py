import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import config 

class PendulumCartEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": int(1/config.DT)
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        high = np.array([
            config.MAX_POSITION * 2, 
            np.finfo(np.float32).max,
            np.pi, 
            np.finfo(np.float32).max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=-config.MAX_VOLTAGE, 
            high=config.MAX_VOLTAGE, 
            shape=(1,), 
            dtype=np.float32
        )

        self.state = None
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        x, x_dot, theta, theta_dot = self.state
        
        voltage = np.clip(action[0], -config.MAX_VOLTAGE, config.MAX_VOLTAGE)
        force = voltage * config.VOLTAGE_TO_FORCE_FACTOR

        total_mass = config.MASS_CART + config.MASS_POLE
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        temp_1 = force + config.MASS_POLE * config.LENGTH * (theta_dot**2) * sin_theta - config.MU_C * x_dot
        num_theta = (total_mass * config.GRAVITY * sin_theta - 
                     cos_theta * temp_1 - 
                     (total_mass * config.MU_P * theta_dot) / (config.MASS_POLE * config.LENGTH))
        den_theta = config.LENGTH * ((4.0/3.0)*total_mass - config.MASS_POLE * (cos_theta**2))
        theta_acc = num_theta / den_theta

        x_acc = (temp_1 - config.MASS_POLE * config.LENGTH * theta_acc * cos_theta) / total_mass

        x = x + config.DT * x_dot
        x_dot = x_dot + config.DT * x_acc
        theta = theta + config.DT * theta_dot
        theta_dot = theta_dot + config.DT * theta_acc

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(x < -config.MAX_POSITION or x > config.MAX_POSITION)
        truncated = bool(self.current_step >= config.MAX_EPISODE_STEPS)

        # ==================================================
        # LA NUEVA FUNCIÓN DE RECOMPENSA (Estrictamente Positiva)
        # ==================================================
        if not terminated:
            # 1. Ángulo: Pasa de [-1, 1] a un rango de [0, 1]. 
            # 1 = Arriba perfecto, 0 = Abajo perfecto.
            reward_theta = (1.0 + math.cos(theta)) / 2.0
            
            # 2. Posición (Gaussiana): Da 1.0 en el centro exacto (x=0) 
            # y cae suavemente a casi 0 cuando te acercas a los límites.
            reward_x = math.exp(-((x / 0.15)**2))
            
            # 3. Bono de supervivencia: Premia simplemente existir sin chocarse.
            # Esto evita el suicidio del agente.
            alive_bonus = 0.5
            
            # 4. Penalización suave de control
            penalty_u = 0.05 * (voltage / config.MAX_VOLTAGE)**2
            
            # El agente ahora gana puntos continuamente. Su objetivo es maximizarlos
            # yendo al centro y subiendo el péndulo.
            reward = reward_theta + reward_x + alive_bonus - penalty_u
        else:
            # Castigo severo que arruina la suma positiva que llevaba
            reward = -50.0 

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # ==================================================
        # NACIMIENTO EN EL LATERAL DEL CARRIL
        # ==================================================
        # Elegimos al azar si nace en el lado izquierdo o derecho (e.g., al 70% del límite del carril)
        lado = self.np_random.choice([-1.0, 1.0])
        posicion_inicial = lado * (config.MAX_POSITION * 0.7) 
        
        self.state = np.array([
            posicion_inicial + self.np_random.uniform(low=-0.02, high=0.02), # x lateral con ruido
            self.np_random.uniform(low=-0.05, high=0.05), # x_dot casi 0
            np.pi + self.np_random.uniform(low=-0.1, high=0.1), # theta hacia abajo
            self.np_random.uniform(low=-0.05, high=0.05)  # theta_dot casi 0
        ], dtype=np.float32)
        
        return self.state, {}

    def render(self):
        screen_width = 600
        screen_height = 400
        world_width = config.MAX_POSITION * 3 
        scale = screen_width / world_width
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else: 
                self.screen = pygame.Surface((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255)) 

        if self.state is None:
            return None

        x = self.state[0]
        theta = self.state[2]

        cart_y = int(screen_height / 2)
        cart_x = int(screen_width / 2 + x * scale)
        
        limit_left = int(screen_width / 2 - config.MAX_POSITION * scale)
        limit_right = int(screen_width / 2 + config.MAX_POSITION * scale)
        pygame.draw.line(self.screen, (255, 0, 0), (limit_left, cart_y - 10), (limit_left, cart_y + 10), 4)
        pygame.draw.line(self.screen, (255, 0, 0), (limit_right, cart_y - 10), (limit_right, cart_y + 10), 4)
        pygame.draw.line(self.screen, (0, 0, 0), (limit_left, cart_y), (limit_right, cart_y), 2)

        cart_width = 50
        cart_height = 30
        pygame.draw.rect(self.screen, (0, 0, 255), (cart_x - cart_width/2, cart_y - cart_height/2, cart_width, cart_height))

        pole_length = config.LENGTH * scale * 2
        end_x = cart_x + pole_length * math.sin(theta)
        end_y = cart_y - pole_length * math.cos(theta)
        pygame.draw.line(self.screen, (0, 255, 0), (cart_x, cart_y), (end_x, end_y), 6)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None