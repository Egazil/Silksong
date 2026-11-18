import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ambiente_silksong import SilksongEnv
from etapas import obter_etapa
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from treinar_silksong import MonitoramentoCompletoCallback, criar_modelo_ppo


EPISODIOS_TREINO = 25
TEMPO_LIMITE_PADRAO_SEGUNDOS = 120.0
PASTA_RESULTADOS = Path("logs_treinamento") / "etapa_03_dar_dano" / "treino_ppo"
PASTA_MODELOS = (
    Path("modelos_etapas")
    / "etapa_03_dar_dano"
    / "treino_ppo"
)


def gerar_grafico_pontuacoes(pontuacoes, caminho_grafico):
    if not pontuacoes:
        print("[Gráfico] Nenhum episódio completo; não há pontuações para plotar.")
        return False

    episodios = np.arange(1, len(pontuacoes) + 1)
    valores = np.asarray(pontuacoes, dtype=float)
    plt.figure(figsize=(10, 5))
    try:
        plt.plot(
            episodios,
            valores,
            marker="o",
            label="Pontuação por episódio",
            color="#1f77b4",
        )
        if len(valores) >= 5:
            janela = min(10, len(valores))
            media_movel = np.convolve(
                valores, np.ones(janela) / janela, mode="valid"
            )
            episodios_media = np.arange(janela, len(valores) + 1)
            plt.plot(
                episodios_media,
                media_movel,
                label=f"Média móvel ({janela} episódios)",
                color="#ff7f0e",
                linewidth=2,
            )

        plt.title("Pontuação PPO por episódio - Etapa 03: Dar Dano")
        plt.xlabel("Episódio")
        plt.ylabel("Pontuação")
        plt.xticks(episodios)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(caminho_grafico, format="png", dpi=300)
    finally:
        plt.close()

    print(f"[Gráfico PNG salvo] {caminho_grafico}")
    return True


def ler_pontuacoes_monitor(caminho_monitor):
    with Path(caminho_monitor).open("r", encoding="utf-8-sig", newline="") as arquivo:
        linhas_csv = (linha for linha in arquivo if not linha.startswith("#"))
        return [
            float(linha["r"])
            for linha in csv.DictReader(linhas_csv)
            if linha.get("r")
        ]


def analisar_argumentos():
    parser = argparse.ArgumentParser(
        description="Treina PPO puro na etapa de dano por um limite de partidas."
    )
    parser.add_argument(
        "--limite-steps",
        type=int,
        default=None,
        help="Limite de steps por partida; por padrão usa o da etapa de dano.",
    )
    parser.add_argument(
        "--limite-segundos",
        type=float,
        default=TEMPO_LIMITE_PADRAO_SEGUNDOS,
        help="Limite de tempo do jogo por partida (padrão: 120 segundos).",
    )
    return parser.parse_args()


def main():
    args = analisar_argumentos()
    etapa = obter_etapa("dano")
    limite_steps = (
        args.limite_steps
        if args.limite_steps is not None
        else etapa.MAX_EPISODE_STEPS
    )
    if limite_steps < 1:
        raise ValueError("--limite-steps precisa ser maior que zero.")
    if args.limite_segundos <= 0:
        raise ValueError("--limite-segundos precisa ser maior que zero.")

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    caminho_modelo = PASTA_MODELOS / "politica_dano_ppo"
    caminho_grafico = PASTA_RESULTADOS / "desempenho_ppo.png"
    caminho_telemetria = PASTA_RESULTADOS / "auditoria_telemetria.jsonl"
    caminho_monitor = PASTA_RESULTADOS / "monitor.csv"
    caminho_checkpoints = PASTA_MODELOS / "checkpoints"
    caminho_checkpoints.mkdir(parents=True, exist_ok=True)

    total_timesteps = 2_147_483_647
    print(f"=== Treino PPO puro: {etapa.TITULO} ===")
    print(
        f"Limite: exatamente {EPISODIOS_TREINO} partidas; cada partida pode durar "
        f"até {limite_steps} steps ou {args.limite_segundos:.0f}s de tempo de jogo."
    )
    print(
        "A Mãe Musgo fica com a FSM ativa e movimento travado; ao chegar a "
        f"{etapa.LIMIAR_HP_CHEFE_EPISODIO} HP, o episódio termina e o reset "
        "restaura a vida da chefe."
    )
    print(
        "A política PPO será inicializada sem pesos genéticos. "
        "O treinamento para somente ao completar o limite de partidas."
    )

    env_base = SilksongEnv(
        etapa="dano",
        host="127.0.0.1",
        port=5001,
        max_episode_steps=limite_steps,
        max_episode_seconds=args.limite_segundos,
        arquivo_log=str(caminho_telemetria),
        preservar_estado_inicial=True,
        penalidade_tempo=0.0,
        penalidade_parede=1.0,
    )
    env = Monitor(env_base, filename=str(caminho_monitor))
    callback_episodios = MonitoramentoCompletoCallback(
        etapa_id=etapa.ID,
        pasta_videos=str(PASTA_RESULTADOS / "videos"),
        caminho_grafico=str(caminho_grafico),
        habilitar_video=False,
        limite_episodios=EPISODIOS_TREINO,
    )
    callback_checkpoint = CheckpointCallback(
        save_freq=max(1, min(4000, total_timesteps)),
        save_path=str(caminho_checkpoints),
        name_prefix="etapa_03_dar_dano_ppo",
    )
    modelo = None

    try:
        modelo = criar_modelo_ppo(env)
        print(
            "Entre na arena, aguarde o mod conectar e pressione ENTER "
            "para iniciar o treinamento PPO."
        )
        input()
        modelo.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=True,
            callback=[callback_episodios, callback_checkpoint],
        )
        print(
            f"\nTreino encerrado após {len(callback_episodios.recompensas_episodios)} "
            "partidas."
        )
    except KeyboardInterrupt:
        print("\nTreino interrompido pelo usuário; salvando o progresso PPO.")
    finally:
        try:
            pontuacoes = callback_episodios.recompensas_episodios
            if not pontuacoes and caminho_monitor.is_file():
                pontuacoes = ler_pontuacoes_monitor(caminho_monitor)
            if pontuacoes:
                print("\n=== PONTUAÇÃO DE CADA EPISÓDIO ===")
                for numero, pontuacao in enumerate(pontuacoes, start=1):
                    print(f"Episódio {numero}: {pontuacao:.2f} pontos")
            else:
                print(
                    "[Pontuação] O monitor não registrou episódios completos; "
                    "não há pontuações para exibir."
                )
            if modelo is not None:
                modelo.save(str(caminho_modelo))
                print(f"Modelo PPO salvo em: {caminho_modelo}.zip")
        finally:
            try:
                gerar_grafico_pontuacoes(
                    pontuacoes,
                    caminho_grafico,
                )
            finally:
                env.close()


if __name__ == "__main__":
    main()
