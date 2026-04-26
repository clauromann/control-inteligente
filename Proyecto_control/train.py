import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder # Nuevas importaciones

import config
from env import PendulumCartEnv

def main():
    print("Inicializando entornos...")
    
    # 1. Entorno de Entrenamiento (Ciego, para ir a máxima velocidad)
    env = PendulumCartEnv(render_mode=None)
    env = Monitor(env, config.LOG_DIR)
    
    # 2. Entorno de Evaluación preparado para Vídeo (Envolvemos en Vectorized Environment)
    # Necesitamos envolverlo en DummyVecEnv para que VecVideoRecorder lo acepte
    eval_env = DummyVecEnv([lambda: Monitor(PendulumCartEnv(render_mode="rgb_array"))])
    
    # 3. El Grabador de Vídeo
    # Grabará un vídeo de 500 pasos (1 episodio entero) cada 10.000 pasos de entrenamiento
    eval_env = VecVideoRecorder(
        eval_env, 
        video_folder=config.VIDEO_DIR,
        record_video_trigger=lambda x: x == 0, # El EvalCallback controlará cuándo grabar
        video_length=config.MAX_EPISODE_STEPS,
        name_prefix="ppo_eval_video"
    )

    # 4. El Callback (Ahora le pasamos el entorno que graba)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config.MODEL_DIR,
        log_path=config.LOG_DIR,
        eval_freq=100000, # Evaluar (y grabar) cada 10.000 pasos
        deterministic=True,
        render=False # Falso porque VecVideoRecorder ya se encarga de guardar el MP4 internamente
    )

    # 5. Configurar el agente PPO
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=config.LOG_DIR # ¡Los vídeos irán a TensorBoard también!
    )

    print("Iniciando el entrenamiento de PPO...")
    total_timesteps = 1_000_000 
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_SwingUp_Run2" # Cambiamos el nombre para no pisar el anterior
    )

    model.save(os.path.join(config.MODEL_DIR, "ppo_final_model"))
    print("¡Entrenamiento finalizado!")

if __name__ == "__main__":
    main()


# import os
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import EvalCallback

# import config
# from env import PendulumCartEnv

# def main():
#     print("Inicializando entornos...")
    
#     # 1. Crear el entorno de entrenamiento
#     # Usamos Monitor para que TensorBoard registre las recompensas y longitudes de episodios correctamente
#     env = PendulumCartEnv(render_mode=None) # Sin render para que entrene a máxima velocidad
#     env = Monitor(env, config.LOG_DIR)

#     # 2. Crear un entorno separado para evaluación
#     # Esto es crucial: queremos evaluar al agente en un entorno limpio sin afectar el entrenamiento
#     eval_env = PendulumCartEnv(render_mode=None)
#     eval_env = Monitor(eval_env)

#     # 3. Configurar el Callback de Evaluación
#     # Cada 10,000 pasos de simulación, el agente se pondrá a prueba.
#     # Si obtiene una recompensa media mejor que las anteriores, se guardará en MODEL_DIR como 'best_model.zip'
#     eval_callback = EvalCallback(
#         eval_env,
#         best_model_save_path=config.MODEL_DIR,
#         log_path=config.LOG_DIR,
#         eval_freq=10000, 
#         deterministic=True, # Usar la política de forma determinista para la evaluación
#         render=False
#     )

#     # 4. Configurar el agente PPO
#     # Ampliamos ligeramente la red neuronal (net_arch=[128, 128]) porque el problema de swing-up 
#     # es más complejo que el CartPole estándar.
#     policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
#     model = PPO(
#         "MlpPolicy",
#         env,
#         learning_rate=3e-4,
#         n_steps=2048, # Pasos por actualización
#         batch_size=64,
#         policy_kwargs=policy_kwargs,
#         verbose=1,
#         tensorboard_log=config.LOG_DIR
#     )

#     # 5. Iniciar el Entrenamiento
#     print("Iniciando el entrenamiento de PPO...")
#     # 1 millón de timesteps es un estándar sólido para asegurar que logra aprender el swing-up y equilibrar
#     total_timesteps = 1_000_000 
    
#     model.learn(
#         total_timesteps=total_timesteps,
#         callback=eval_callback,
#         tb_log_name="PPO_SwingUp_Run1"
#     )

#     # 6. Guardar el modelo del final del entrenamiento
#     model.save(os.path.join(config.MODEL_DIR, "ppo_final_model"))
#     print("¡Entrenamiento finalizado con éxito!")

# if __name__ == "__main__":
#     main()