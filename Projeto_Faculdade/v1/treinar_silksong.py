import os
import matplotlib.pyplot as plt
import numpy as np
from ambiente_silksong import SilksongEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Tentativa de importação de bibliotecas de gravação de vídeo (opcional)
GRAVADOR_DISPONIVEL = False
try:
    import mss
    import cv2
    GRAVADOR_DISPONIVEL = True
except ImportError:
    print("[Aviso] Bibliotecas 'mss' ou 'opencv-python' não encontradas.")
    print("Para habilitar a gravação automática de vídeos em .mp4, instale com: pip install mss opencv-python\n")

print("=== TREINAMENTO DE RL PARA SILKSONG COM GRAVAÇÃO INTELIGENTE & MONITORAMENTO ===")

NOME_MODELO = "modelo_silksong_ppo"
PASTA_CHECKPOINTS = "modelos_checkpoint"
PASTA_LOGS = "logs_treinamento"
PASTA_VIDEOS = "videos_treinamento"

os.makedirs(PASTA_CHECKPOINTS, exist_ok=True)
os.makedirs(PASTA_LOGS, exist_ok=True)
os.makedirs(PASTA_VIDEOS, exist_ok=True)


# --- CALLBACK PARA ACOMPANHAMENTO DE PESOS, EPISÓDIOS E GRAVAÇÃO DE VÍDEOS ---
class MonitoramentoCompletoCallback(BaseCallback):
    def __init__(self, verbose=0, habilitar_video=True):
        super(MonitoramentoCompletoCallback, self).__init__(verbose)
        self.recompensas_episodios = []
        self.duracao_episodios = []
        self.total_atualizacoes = 0
        self.melhor_recompensa = -float("inf")

        # Configurações do gravador
        self.habilitar_video = habilitar_video and GRAVADOR_DISPONIVEL
        self.buffer_frames = []
        self.sct = None
        self.monitor_area = None

        if self.habilitar_video:
            try:
                self.sct = mss.mss()
                # Captura a tela principal (monitor 1)
                self.monitor_area = self.sct.monitors[1]
                print(f"🎥 [Gravador de Vídeo Ativo] Clipes marcantes e recordes serão salvos em: '{PASTA_VIDEOS}/'")
            except Exception as e:
                print(f"[Aviso] Falha ao inicializar MSS: {e}. Gravação de vídeo desligada.")
                self.habilitar_video = False

    def _salvar_video(self, caminho_arquivo, fps=20):
        """Salva a lista de frames em buffer como um arquivo MP4 leve (720p)."""
        if not self.buffer_frames:
            return

        try:
            altura, largura, _ = self.buffer_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(caminho_arquivo, fourcc, fps, (largura, altura))

            for frame in self.buffer_frames:
                out.write(frame)

            out.release()
            print(f"🎬 [VÍDEO GRAVADO COM SUCESSO!] Salvo em: {caminho_arquivo}")
        except Exception as e:
            print(f"[Erro ao codificar vídeo]: {e}")

    def _on_step(self) -> bool:
        # 1. Se gravação ativa, captura o frame atual e guarda na memória RAM
        if self.habilitar_video and self.sct:
            try:
                screenshot = self.sct.grab(self.monitor_area)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame = cv2.resize(frame, (1280, 720)) # Resolução 720p leve
                self.buffer_frames.append(frame)
            except Exception:
                pass

        # 2. Checa se o episódio encerrou neste step
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.recompensas_episodios.append(ep_reward)
                self.duracao_episodios.append(ep_length)
                ep_num = len(self.recompensas_episodios)

                # Critérios de Gravação de Vídeo:
                # A: Marcos Históricos (Ep 1, 5, 10, 25, 50, 100, 200, 500...)
                marcos = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
                eh_marco = (ep_num in marcos) or (ep_num % 100 == 0)

                # B: Novo Recorde de Pontuação
                eh_recorde = ep_reward > self.melhor_recompensa
                if eh_recorde:
                    self.melhor_recompensa = ep_reward

                print(f"📊 [FIM DE EPISÓDIO {ep_num}] Recompensa: {ep_reward:.2f} | Duração: {ep_length} passos" +
                      (" 🏆 [NOVO RECORDE!]" if eh_recorde else ""))

                # Salva o vídeo se cumprir algum dos critérios
                if self.habilitar_video:
                    if eh_recorde and ep_num > 1:
                        caminho = os.path.join(PASTA_VIDEOS, f"RECORDE_ep_{ep_num:04d}_pontos_{ep_reward:.1f}.mp4")
                        self._salvar_video(caminho)
                    elif eh_marco:
                        caminho = os.path.join(PASTA_VIDEOS, f"marco_ep_{ep_num:04d}_pontos_{ep_reward:.1f}.mp4")
                        self._salvar_video(caminho)

                    # Limpa o buffer da memória RAM após a decisão
                    self.buffer_frames.clear()

        return True

    def _on_rollout_end(self) -> bool:
        self.total_atualizacoes += 1
        print("\n" + "🔥" * 65)
        print(f" 🧠 [CICLO {self.total_atualizacoes} - PESOS ATUALIZADOS!] O PPO otimizou a rede neural.")
        print("🔥" * 65 + "\n")
        return True

    def gerar_grafico(self, caminho_salvar="desempenho_treinamento.png"):
        """Gera e salva um gráfico de linha com a evolução do agente ao longo dos episódios."""
        if not self.recompensas_episodios:
            print("[Gráfico] Nenhum episódio completo registrado para gerar gráfico.")
            return

        episodios = np.arange(1, len(self.recompensas_episodios) + 1)
        recompensas = np.array(self.recompensas_episodios)

        plt.figure(figsize=(10, 5))
        plt.plot(episodios, recompensas, label="Recompensa por Episódio", color="#1f77b4", alpha=0.4)

        if len(recompensas) >= 5:
            janela = min(10, len(recompensas))
            media_movel = np.convolve(recompensas, np.ones(janela)/janela, mode='valid')
            x_media = np.arange(janela, len(recompensas) + 1)
            plt.plot(x_media, media_movel, label=f"Tendência (Média Móvel {janela} eps)", color="#ff7f0e", linewidth=2.5)

        plt.title("Evolução do Treinamento - Silksong RL (PPO)", fontsize=14, fontweight="bold")
        plt.xlabel("Episódios", fontsize=12)
        plt.ylabel("Recompensa Total", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        plt.savefig(caminho_salvar, dpi=300)
        plt.close()
        print(f"\n📈 [Gráfico Salvo com Sucesso!] Verifique o arquivo: '{caminho_salvar}'")


# 1. Cria o ambiente e envolve com o Monitor para métricas precisas
env_base = SilksongEnv(host="127.0.0.1", port=5001)
env = Monitor(env_base, filename=os.path.join(PASTA_LOGS, "monitor.csv"))

# 2. Configura o modelo PPO mantendo a arquitetura leve padrão (64x64, Tanh)
model = None
if os.path.exists(f"{NOME_MODELO}.zip"):
    print(f"Tentando carregar pesos salvos anteriormente de '{NOME_MODELO}.zip'...")
    try:
        model = PPO.load(NOME_MODELO, env=env)
        print("Modelo anterior carregado com sucesso!")
    except Exception as e:
        print(f"Aviso: Não foi possível carregar os pesos antigos devido à evolução da arquitetura (26 entradas e MultiBinary): {e}")
        print("Iniciando um novo modelo do zero adaptado perfeitamente para a V2...")
        model = None

if model is None:
    print("Iniciando IA do zero (Arquitetura padrão [64, 64], Tanh, 26 observações, MultiBinary)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=400,      # Coleta exatamente 400 steps por ciclo de atualização
        batch_size=50,    # 400 / 50 = 8 mini-batches bem distribuídos
        gamma=0.99,
        gae_lambda=0.95,
    )

checkpoint_callback = CheckpointCallback(
    save_freq=4000,
    save_path=PASTA_CHECKPOINTS,
    name_prefix="silksong_ppo",
)
monitoramento_callback = MonitoramentoCompletoCallback(habilitar_video=True)

print("\n" + "=" * 65)
print("🛑 STATUS: CONECTADO COM SUCESSO E EM SILÊNCIO!")
print("Instruções:")
print("1. Volte para o jogo e controle a Hornet manualmente.")
print("2. Caminhe até a arena da Mãe Musgo.")
print("3. Quando estiver pronto na arena, volte aqui e pressione ENTER.")
print("=" * 65 + "\n")

input("👉 Pressione [ENTER] para que a IA assuma os controles e inicie o treinamento...")

print("\n--- TREINAMENTO INICIADO (400 STEPS POR ATUALIZAÇÃO) ---")
print("Pressione Ctrl+C a qualquer momento para pausar, salvar e gerar o gráfico PNG.\n")

try:
    model.learn(
        total_timesteps=100000,
        reset_num_timesteps=False,
        callback=[monitoramento_callback, checkpoint_callback],
    )

    model.save(NOME_MODELO)
    print(f"\n[Sucesso] Treinamento concluído e salvo em '{NOME_MODELO}.zip'!")

except KeyboardInterrupt:
    print("\n[Aviso] Interrompido pelo usuário. Salvando pesos atuais...")
    model.save(NOME_MODELO)
    print("[Sucesso] Progresso salvo com segurança.")

finally:
    # Gera o gráfico de desempenho automaticamente
    monitoramento_callback.gerar_grafico("desempenho_treinamento.png")
    env.close()
