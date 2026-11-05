import os
from ambiente_silksong import SilksongEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

print("=== TREINAMENTO DE RL PARA SILKSONG COM FEEDBACK EM TEMPO REAL ===")

NOME_MODELO = "modelo_silksong_ppo"
PASTA_CHECKPOINTS = "modelos_checkpoint"
os.makedirs(PASTA_CHECKPOINTS, exist_ok=True)


# --- INDICADOR DE ATUALIZAÇÃO DE PESOS DA REDE NEURAL ---
class IndicadorAtualizacaoPesos(BaseCallback):

  def __init__(self, verbose=0):
    super(IndicadorAtualizacaoPesos, self).__init__(verbose)

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> bool:
    print("\n" + "🔥" * 60)
    print(
        " 🧠 [PESOS ATUALIZADOS!] O PPO otimizou a rede neural com os dados da"
        " arena."
    )
    print("🔥" * 60 + "\n")
    return True


env = SilksongEnv(host="127.0.0.1", port=5001)

if os.path.exists(f"{NOME_MODELO}.zip"):
  print(
      f"Carregando pesos salvos anteriormente de '{NOME_MODELO}.zip' para"
      " continuar..."
  )
  model = PPO.load(NOME_MODELO, env=env)
else:
  print("Nenhum modelo anterior encontrado. Iniciando IA do zero...")
  model = PPO(
      "MlpPolicy",
      env,
      verbose=1,
      learning_rate=0.0003,
      n_steps=100,  # Atualiza os pesos a cada 512 passos (muito mais ágil!)
      batch_size=64,
  )

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=PASTA_CHECKPOINTS,
    name_prefix="silksong_ppo",
)
indicador_callback = IndicadorAtualizacaoPesos()

print("\n--- TREINAMENTO INICIADO ---")
print("Pressione Ctrl+C a qualquer momento para pausar e salvar.\n")

try:
  model.learn(
      total_timesteps=50000,
      reset_num_timesteps=False,
      callback=[indicador_callback, checkpoint_callback],
  )

  model.save(NOME_MODELO)
  print(f"\n[Sucesso] Treinamento concluído e salvo em '{NOME_MODELO}.zip'!")

except KeyboardInterrupt:
  print("\n[Aviso] Interrompido pelo usuário. Salvando pesos atuais...")
  model.save(NOME_MODELO)
  print("[Sucesso] Progresso salvo com segurança.")

finally:
  env.close()