import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from ambiente_silksong import SilksongEnv
from etapas import ORDEM_ETAPAS, obter_etapa
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

# --- CALLBACK PARA ACOMPANHAMENTO DE PESOS, EPISÓDIOS E GRAVAÇÃO DE VÍDEOS ---
class MonitoramentoCompletoCallback(BaseCallback):
    def __init__(
        self,
        etapa_id,
        pasta_videos,
        caminho_grafico,
        verbose=0,
        habilitar_video=True,
        limite_episodios=None,
    ):
        super(MonitoramentoCompletoCallback, self).__init__(verbose)
        self.etapa_id = etapa_id
        self.pasta_videos = pasta_videos
        self.caminho_grafico = caminho_grafico
        self.limite_episodios = limite_episodios
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
                print(
                    f"🎥 [Gravador de Vídeo Ativo] Clipes da etapa serão salvos em: "
                    f"'{self.pasta_videos}/'"
                )
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

                print(
                    f"[FIM DE EPISÓDIO {ep_num}] Pontuação: {ep_reward:.2f} | "
                    f"Duração: {ep_length} passos"
                    + (" [NOVO RECORDE]" if eh_recorde else "")
                )

                # Salva o vídeo se cumprir algum dos critérios
                if self.habilitar_video:
                    if eh_recorde and ep_num > 1:
                        caminho = os.path.join(
                            self.pasta_videos,
                            f"RECORDE_ep_{ep_num:04d}_pontos_{ep_reward:.1f}.mp4",
                        )
                        self._salvar_video(caminho)
                    elif eh_marco:
                        caminho = os.path.join(
                            self.pasta_videos,
                            f"marco_ep_{ep_num:04d}_pontos_{ep_reward:.1f}.mp4",
                        )
                        self._salvar_video(caminho)

                    # Limpa o buffer da memória RAM após a decisão
                    self.buffer_frames.clear()

                if (
                    self.limite_episodios is not None
                    and ep_num >= self.limite_episodios
                ):
                    print(
                        f"[Treino] Limite de {self.limite_episodios} episódios "
                        "concluído; encerrando o PPO."
                    )
                    return False

        return True

    def _on_rollout_end(self) -> bool:
        self.total_atualizacoes += 1
        print("\n" + "🔥" * 65)
        print(f" 🧠 [CICLO {self.total_atualizacoes} - PESOS ATUALIZADOS!] O PPO otimizou a rede neural.")
        print("🔥" * 65 + "\n")
        return True

    def gerar_grafico(self):
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

        plt.title(
            f"Evolução do Treinamento - {self.etapa_id} (PPO)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Episódios", fontsize=12)
        plt.ylabel("Recompensa Total", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        plt.savefig(self.caminho_grafico, dpi=300)
        plt.close()
        print(
            f"\n[Gráfico salvo com sucesso] Verifique o arquivo: "
            f"'{self.caminho_grafico}'"
        )


def analisar_argumentos():
    parser = argparse.ArgumentParser(
        description="Treinamento PPO por etapas do currículo de combate."
    )
    parser.add_argument(
        "--etapa",
        choices=ORDEM_ETAPAS,
        default="curar",
        help="Etapa a treinar: curar, sobreviver, dano ou vencer.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Quantidade de passos de treinamento para esta execução.",
    )
    parser.add_argument(
        "--limite-steps",
        type=int,
        default=None,
        help="Limite de passos por episódio (o padrão vem do arquivo da etapa).",
    )
    parser.add_argument(
        "--modelo-inicial",
        default=None,
        help="Modelo PPO inicial explícito; substitui a seleção automática.",
    )
    parser.add_argument(
        "--reiniciar-etapa",
        action="store_true",
        help="Ignora o modelo salvo desta etapa e inicia pelo modelo da etapa anterior.",
    )
    parser.add_argument(
        "--modelo-genetico",
        default=None,
        help="Arquivo .npz com a melhor política evoluída para transferir ao PPO.",
    )
    return parser.parse_args()


def selecionar_modelo_inicial(args, etapa):
    pasta_etapas = "modelos_etapas"
    modelo_atual = os.path.join(pasta_etapas, f"{etapa.ID}.zip")
    if args.modelo_inicial:
        if not os.path.isfile(args.modelo_inicial):
            raise FileNotFoundError(
                f"Modelo inicial não encontrado: {args.modelo_inicial}"
            )
        return args.modelo_inicial

    if (
        os.path.isfile(modelo_atual)
        and not args.reiniciar_etapa
        and not args.modelo_genetico
    ):
        return modelo_atual

    indice_etapa = ORDEM_ETAPAS.index(args.etapa)
    if indice_etapa > 0:
        etapa_anterior = obter_etapa(ORDEM_ETAPAS[indice_etapa - 1])
        modelo_anterior = os.path.join(pasta_etapas, f"{etapa_anterior.ID}.zip")
        if not os.path.isfile(modelo_anterior):
            raise FileNotFoundError(
                f"A etapa '{etapa_anterior.ID}' precisa ser treinada primeiro. "
                f"Modelo esperado: {modelo_anterior}"
            )
        return modelo_anterior
    return None


def criar_modelo_ppo(env):
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=400,
        batch_size=50,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )


def main():
    args = analisar_argumentos()
    if args.timesteps < 1:
        raise ValueError("--timesteps precisa ser maior que zero.")
    if args.modelo_genetico and args.etapa != "dano":
        raise ValueError(
            "--modelo-genetico só pode ser aplicado ao treino da etapa 'dano'."
        )
    if args.modelo_genetico and not os.path.isfile(args.modelo_genetico):
        raise FileNotFoundError(f"Genoma não encontrado: {args.modelo_genetico}")

    etapa = obter_etapa(args.etapa)
    limite_steps = (
        args.limite_steps
        if args.limite_steps is not None
        else etapa.MAX_EPISODE_STEPS
    )
    if limite_steps < 1:
        raise ValueError("--limite-steps precisa ser maior que zero.")

    pasta_modelos = "modelos_etapas"
    pasta_logs = os.path.join("logs_treinamento", etapa.ID)
    pasta_checkpoints = os.path.join(pasta_modelos, "checkpoints", etapa.ID)
    pasta_videos = os.path.join("videos_treinamento", etapa.ID)
    for pasta in (pasta_logs, pasta_checkpoints, pasta_videos):
        os.makedirs(pasta, exist_ok=True)

    caminho_modelo = os.path.join(pasta_modelos, etapa.ID)
    caminho_grafico = os.path.join(pasta_logs, "desempenho.png")
    caminho_telemetria = os.path.join(pasta_logs, "auditoria_telemetria.jsonl")
    caminho_monitor = os.path.join(pasta_logs, "monitor.csv")
    modelo_inicial = selecionar_modelo_inicial(args, etapa)

    print(
        f"=== TREINAMENTO: {etapa.ID} - {etapa.TITULO} "
        f"({args.timesteps} timesteps) ==="
    )
    env_base = SilksongEnv(
        etapa=args.etapa,
        host="127.0.0.1",
        port=5001,
        max_episode_steps=limite_steps,
        arquivo_log=caminho_telemetria,
    )
    env = Monitor(env_base, filename=caminho_monitor)
    monitoramento_callback = MonitoramentoCompletoCallback(
        etapa_id=etapa.ID,
        pasta_videos=pasta_videos,
        caminho_grafico=caminho_grafico,
        habilitar_video=True,
    )
    model = None

    try:
        if modelo_inicial:
            print(f"Carregando política inicial: {modelo_inicial}")
            modelo_origem = PPO.load(modelo_inicial, device="cpu")
            espaco_obs_origem = modelo_origem.observation_space.shape
            espaco_acoes_origem = modelo_origem.action_space.shape
            if espaco_obs_origem not in {(26,), (37,)}:
                raise ValueError(
                    f"Modelo inicial usa observações {espaco_obs_origem}; "
                    "só há migração definida para 26 ou 37 valores."
                )
            if espaco_acoes_origem not in {(6,), (7,)}:
                raise ValueError(
                    f"Modelo inicial usa ações {espaco_acoes_origem}; "
                    "só há migração definida para 6 ou 7 ações."
                )

            if (
                modelo_origem.observation_space == env.observation_space
                and modelo_origem.action_space == env.action_space
            ):
                model = PPO.load(modelo_inicial, env=env)
            else:
                model = criar_modelo_ppo(env)
                from genetico import transferir_pesos_ppo

                transferidos = transferir_pesos_ppo(modelo_origem, model)
                if not transferidos:
                    raise RuntimeError(
                        f"Nenhum peso compatível pôde ser transferido de "
                        f"'{modelo_inicial}'."
                    )
                print(
                    f"Pesos migrados para a rede atual "
                    f"({len(transferidos)} tensores)."
                )
        else:
            print("Nenhum estágio anterior encontrado; inicializando PPO do zero.")
            model = criar_modelo_ppo(env)

        if args.modelo_genetico:
            from genetico import carregar_genoma, transferir_genoma_para_ppo

            genoma, metadados = carregar_genoma(args.modelo_genetico)
            transferir_genoma_para_ppo(model, genoma)
            print(
                f"Política genética transferida: geração {metadados['geracao']}, "
                f"agente {metadados['agente']}, pontuação "
                f"{metadados['pontuacao']:.3f}."
            )

        checkpoint_callback = CheckpointCallback(
            save_freq=4000,
            save_path=pasta_checkpoints,
            name_prefix=etapa.ID,
        )

        print("\n" + "=" * 65)
        print(f"Etapa ativa: {etapa.TITULO}")
        print("1. Volte ao jogo e entre na arena.")
        print("2. Pressione ENTER aqui para iniciar o treinamento.")
        print("=" * 65 + "\n")
        input("👉 Pressione [ENTER] para iniciar...")

        model.learn(
            total_timesteps=args.timesteps,
            reset_num_timesteps=False,
            callback=[monitoramento_callback, checkpoint_callback],
        )
        model.save(caminho_modelo)
        print(f"\n[Sucesso] Modelo salvo em '{caminho_modelo}.zip'.")
        if etapa.PROXIMA_ETAPA:
            print(
                f"Quando esta etapa estiver suficiente, treine a próxima com: "
                f"python treinar_silksong.py --etapa {etapa.PROXIMA_ETAPA}"
            )
    except KeyboardInterrupt:
        print("\n[Aviso] Interrompido pelo usuário.")
        if model is not None:
            model.save(caminho_modelo)
            print(f"[Sucesso] Progresso salvo em '{caminho_modelo}.zip'.")
    finally:
        monitoramento_callback.gerar_grafico()
        env.close()


if __name__ == "__main__":
    main()
