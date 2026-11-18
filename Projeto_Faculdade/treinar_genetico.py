import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ambiente_silksong import SilksongEnv
from etapas import obter_etapa
from genetico import (
    DESVIO_MUTACAO,
    GERACOES,
    POPULACAO,
    TAXA_MUTACAO,
    calcular_acao,
    carregar_genoma,
    criar_populacao_inicial,
    criar_proxima_geracao,
    genoma_de_policy,
    salvar_genoma,
    transferir_pesos_ppo,
)
from stable_baselines3 import PPO


TEMPO_LIMITE_AVALIACAO_SEGUNDOS = 120.0


def analisar_argumentos():
    parser = argparse.ArgumentParser(
        description=(
            "Evolui uma política para causar dano à Mãe Musgo parada e exporta "
            "o melhor genoma para o PPO."
        )
    )
    parser.add_argument(
        "--modelo-inicial",
        default=None,
        help=(
            "PPO para inicializar a população. Por padrão, usa o modelo da etapa "
            "de sobrevivência se ele existir."
        ),
    )
    parser.add_argument(
        "--limite-steps",
        type=int,
        default=None,
        help="Limite de steps por avaliação; por padrão usa o valor da etapa de dano.",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="Semente reprodutível para seleção, cruzamento e mutação.",
    )
    return parser.parse_args()


def criar_modelo_referencia(env):
    return PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=400,
        batch_size=50,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
    )


def obter_modelo_inicial(args):
    if args.modelo_inicial:
        if not os.path.isfile(args.modelo_inicial):
            raise FileNotFoundError(
                f"Modelo inicial não encontrado: {args.modelo_inicial}"
            )
        return args.modelo_inicial

    caminho_sobrevivencia = os.path.join(
        "modelos_etapas", "etapa_02_sobreviver.zip"
    )
    if os.path.isfile(caminho_sobrevivencia):
        return caminho_sobrevivencia
    return None


def avaliar_agente(env, genoma, limite_steps):
    observacao, _ = env.reset()
    pontuacao = 0.0
    passos = 0

    for passos in range(1, limite_steps + 1):
        acao = calcular_acao(genoma, observacao)
        observacao, recompensa, terminado, truncado, _ = env.step(acao)
        pontuacao += float(recompensa)
        if terminado or truncado:
            break
    else:
        raise RuntimeError(
            "A avaliação ultrapassou o limite configurado sem terminar o episódio."
        )

    return pontuacao, passos


def gerar_grafico_evolucao(
    pontuacoes_por_geracao: list[list[float]],
    caminho: Path,
) -> None:
    geracoes = np.arange(1, len(pontuacoes_por_geracao) + 1)
    melhores = [max(pontuacoes) for pontuacoes in pontuacoes_por_geracao]
    medias = [float(np.mean(pontuacoes)) for pontuacoes in pontuacoes_por_geracao]
    piores = [min(pontuacoes) for pontuacoes in pontuacoes_por_geracao]
    melhores_acumuladas = np.maximum.accumulate(melhores)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(geracoes, melhores, marker="o", label="Melhor da geração")
    ax.plot(geracoes, medias, marker="s", label="Média da geração")
    ax.plot(geracoes, piores, marker="v", label="Pior da geração")
    ax.plot(
        geracoes,
        melhores_acumuladas,
        linestyle="--",
        label="Melhor acumulado",
    )
    ax.set_title("Evolução do algoritmo genético — etapa de dano")
    ax.set_xlabel("Geração")
    ax.set_ylabel("Pontuação acumulada")
    ax.set_xticks(geracoes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)


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

    pasta_modelos = Path("modelos_etapas") / etapa.ID / "genetico"
    pasta_logs = Path("logs_treinamento") / etapa.ID / "genetico"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_logs.mkdir(parents=True, exist_ok=True)
    caminho_melhor = pasta_modelos / "melhor_policy_genetica.npz"
    caminho_resultados = pasta_logs / "avaliacoes.csv"
    caminho_grafico = pasta_logs / "evolucao_genetica.png"

    print(
        f"=== ALGORITMO GENÉTICO: {GERACOES} gerações, "
        f"{POPULACAO} agentes por geração ==="
    )
    print(
        f"Cada avaliação terá até {limite_steps} steps; "
        f"ou {TEMPO_LIMITE_AVALIACAO_SEGUNDOS:.0f}s de tempo do jogo."
    )
    print(
        "Operadores: elitismo do melhor, crossover uniforme entre os dois melhores, "
        f"mutação Gaussiana ({TAXA_MUTACAO:.0%}, desvio {DESVIO_MUTACAO:.2f})."
    )

    env = SilksongEnv(
        etapa="dano",
        host="127.0.0.1",
        port=5001,
        max_episode_steps=limite_steps,
        max_episode_seconds=TEMPO_LIMITE_AVALIACAO_SEGUNDOS,
        arquivo_log=str(pasta_logs / "auditoria_telemetria.jsonl"),
    )
    melhor_pontuacao = -float("inf")
    melhor_metadado = None

    try:
        modelo_referencia = criar_modelo_referencia(env)
        caminho_inicial = obter_modelo_inicial(args)
        if caminho_inicial:
            print(f"Usando política da etapa anterior: {caminho_inicial}")
            modelo_anterior = PPO.load(caminho_inicial, device="cpu")
            espaco_obs = modelo_anterior.observation_space.shape
            espaco_acoes = modelo_anterior.action_space.shape
            if espaco_obs not in {(26,), (37,)}:
                raise ValueError(
                    f"Modelo inicial usa observações {espaco_obs}; esperado 26 ou 37."
                )
            if espaco_acoes not in {(6,), (7,)}:
                raise ValueError(
                    f"Modelo inicial usa ações {espaco_acoes}; esperado 6 ou 7."
                )
            transferidos = transferir_pesos_ppo(modelo_anterior, modelo_referencia)
            if not transferidos:
                raise RuntimeError(
                    f"Nenhum peso compatível pôde ser transferido de {caminho_inicial}."
                )
            print(f"Pesos compatíveis carregados: {len(transferidos)} tensores.")
            genoma_base = genoma_de_policy(
                modelo_referencia.policy.state_dict(),
                np.random.default_rng(args.semente),
            )
        else:
            print(
                "Modelo da etapa de sobrevivência não encontrado; "
                "iniciando os genomas aleatoriamente."
            )
            genoma_base = None

        rng = np.random.default_rng(args.semente)
        populacao = criar_populacao_inicial(rng, genoma_base)
        with caminho_resultados.open("w", newline="", encoding="utf-8") as arquivo_csv:
            escritor = csv.DictWriter(
                arquivo_csv,
                fieldnames=("geracao", "agente", "pontuacao", "steps"),
            )
            escritor.writeheader()
            pontuacoes_por_geracao = []

            print(
                "Entre na arena; na etapa de dano, o mod posicionará a chefe "
                "no centro e a manterá parada."
            )
            input("Pressione ENTER para iniciar as avaliações genéticas...")

            for geracao in range(1, GERACOES + 1):
                avaliados = []
                pontuacoes_geracao = []
                print(f"\n--- Geração {geracao}/{GERACOES} ---")
                for indice, genoma in enumerate(populacao, start=1):
                    pontuacao, passos = avaliar_agente(env, genoma, limite_steps)
                    avaliados.append((pontuacao, genoma))
                    pontuacoes_geracao.append(pontuacao)
                    escritor.writerow(
                        {
                            "geracao": geracao,
                            "agente": indice,
                            "pontuacao": f"{pontuacao:.6f}",
                            "steps": passos,
                        }
                    )
                    arquivo_csv.flush()
                    print(
                        f"Agente {indice}/{POPULACAO}: "
                        f"pontuação={pontuacao:.3f}, steps={passos}"
                    )

                    if pontuacao > melhor_pontuacao:
                        melhor_pontuacao = pontuacao
                        melhor_metadado = (geracao, indice)
                        salvar_genoma(
                            str(caminho_melhor),
                            genoma,
                            geracao,
                            indice,
                            pontuacao,
                        )

                pontuacoes_por_geracao.append(pontuacoes_geracao)
                gerar_grafico_evolucao(pontuacoes_por_geracao, caminho_grafico)
                print(f"Gráfico atualizado: {caminho_grafico}")

                if geracao < GERACOES:
                    populacao = criar_proxima_geracao(avaliados, rng)

        if melhor_metadado is None:
            raise RuntimeError("Nenhum agente genético foi avaliado.")
        _, metadados = carregar_genoma(str(caminho_melhor))
        print(
            f"\nMelhor política: geração {metadados['geracao']}, "
            f"agente {metadados['agente']}, pontuação "
            f"{metadados['pontuacao']:.3f}."
        )
        print(f"Genoma salvo em: {caminho_melhor}")
        print(
            "Transfira para PPO com: "
            "python treinar_silksong.py --etapa dano "
            f"--modelo-genetico \"{caminho_melhor}\""
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
