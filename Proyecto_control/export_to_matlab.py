import os
import torch
from stable_baselines3 import PPO

import config

# Creamos una clase "envoltorio" (wrapper) aislar solo lo que Simulink necesita
class OnnxablePolicy(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, observation):
        # 1. Pedir a la política la acción determinista (sin ruido de exploración)
        # En PPO, _predict devuelve la acción "en crudo" desde la capa lineal final.
        action = self.model.policy._predict(observation, deterministic=True)
        
        # 2. PPO asume que las acciones base están en el rango [-1, 1] y luego las recorta.
        # Replicamos ese recorte (clip) explícitamente para que quede embebido en el ONNX.
        action_clipped = torch.clamp(action, min=-1.0, max=1.0)
        
        # 3. Desescalar la acción al rango de nuestro sistema ([-0.9, 0.9] Voltios)
        # De esta forma, el bloque de Simulink escupirá directamente los Voltios correctos.
        action_real = action_clipped * config.MAX_VOLTAGE
        
        return action_real

def main():
    model_path = os.path.join(config.MODEL_DIR, "best_model.zip")
    onnx_path = os.path.join(config.MODEL_DIR, "ppo_actor.onnx")

    if not os.path.exists(model_path):
        print(f"Error: No se ha encontrado el modelo en {model_path}")
        return

    print("Cargando el modelo PPO entrenado...")
    # Forzamos la carga en CPU (Simulink inferirá en la CPU por lo general)
    model = PPO.load(model_path, device="cpu")

    # Instanciar nuestro modelo limpio solo con el Actor
    onnxable_model = OnnxablePolicy(model)

    # Crear una observación ficticia para hacer un "trazado" (tracing) de la red neuronal.
    # Tenemos 4 variables de estado: [x, x_dot, theta, theta_dot]
    # El tamaño (1, 4) representa 1 ejemplo en el batch (lote) con 4 características.
    dummy_observation = torch.randn(1, 4)

    print("Exportando la red neuronal a formato ONNX...")
    
    # Exportación oficial de PyTorch a ONNX
    torch.onnx.export(
        onnxable_model,             # El modelo limpio
        dummy_observation,          # Entrada de ejemplo para que detecte las dimensiones
        onnx_path,                  # Ruta de salida
        export_params=True,         # Guardar los pesos entrenados
        opset_version=11,           # Versión estándar muy compatible con MATLAB
        do_constant_folding=True,   # Optimizar el modelo
        input_names=['observation'],# Nombre del puerto de entrada para Simulink
        output_names=['voltage'],   # Nombre del puerto de salida para Simulink
        dynamic_axes={
            'observation': {0: 'batch_size'}, # Permite procesar varios estados a la vez si fuera necesario
            'voltage': {0: 'batch_size'}
        }
    )

    print(f"¡Exportación exitosa! El archivo está listo para MATLAB en: {onnx_path}")

if __name__ == "__main__":
    main()