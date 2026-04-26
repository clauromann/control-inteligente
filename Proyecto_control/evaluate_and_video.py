import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

import config
from env import PendulumCartEnv

def main():
    model_path = os.path.join(config.MODEL_DIR, "best_model.zip")
    
    # Comprobar que el modelo existe antes de intentar cargarlo
    if not os.path.exists(model_path):
        print(f"Error: No se ha encontrado el modelo en {model_path}")
        print("Asegúrate de haber ejecutado train.py primero.")
        return

    print("Cargando el mejor modelo entrenado...")
    model = PPO.load(model_path)

    # 1. Crear el entorno en modo 'rgb_array' (Vital para WSL)
    env = PendulumCartEnv(render_mode="rgb_array")

    # 2. Envolver el entorno con RecordVideo
    # Esto grabará automáticamente episodios y los guardará como .mp4
    # name_prefix le da un nombre base a los archivos generados
    env = RecordVideo(
        env, 
        video_folder=config.VIDEO_DIR,
        name_prefix="swingup_agent",
        episode_trigger=lambda x: True # Grabar TODOS los episodios que ejecutemos aquí
    )

    print(f"Iniciando evaluación. Los vídeos se guardarán en: {config.VIDEO_DIR}")

    # Vamos a grabar 3 episodios distintos para ver cómo se comporta
    num_episodes = 3

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        while not done:
            # El agente decide la acción basándose en la observación actual
            # deterministic=True es importante en evaluación para usar la política óptima aprendida, 
            # sin la exploración aleatoria que se usa durante el entrenamiento.
            action, _states = model.predict(obs, deterministic=True)
            
            # Ejecutar la acción en el entorno
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # El episodio termina si se sale de la pista (terminated) 
            # o si alcanza el límite de tiempo (truncated)
            done = terminated or truncated

        print(f"Episodio {ep + 1} completado: {step_count} pasos | Recompensa Total: {total_reward:.2f}")

    # Cerrar el entorno asegura que el archivo .mp4 se guarde y cierre correctamente
    env.close()
    print("¡Evaluación finalizada! Revisa la carpeta de vídeos.")

if __name__ == "__main__":
    main()