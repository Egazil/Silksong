from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch


OBSERVACAO_DIM = 37
ACOES_DIM = 7
NEURONIOS_OCULTOS = 64
GERACOES = 5
POPULACAO = 3
TAXA_MUTACAO = 0.01
DESVIO_MUTACAO = 0.05

CHAVES_GENOMA = {
    "camada1_pesos": (NEURONIOS_OCULTOS, OBSERVACAO_DIM),
    "camada1_vies": (NEURONIOS_OCULTOS,),
    "camada2_pesos": (NEURONIOS_OCULTOS, NEURONIOS_OCULTOS),
    "camada2_vies": (NEURONIOS_OCULTOS,),
    "saida_pesos": (ACOES_DIM, NEURONIOS_OCULTOS),
    "saida_vies": (ACOES_DIM,),
}

PARAMETROS_POLICY = {
    "camada1_pesos": "mlp_extractor.policy_net.0.weight",
    "camada1_vies": "mlp_extractor.policy_net.0.bias",
    "camada2_pesos": "mlp_extractor.policy_net.2.weight",
    "camada2_vies": "mlp_extractor.policy_net.2.bias",
    "saida_pesos": "action_net.weight",
    "saida_vies": "action_net.bias",
}

INDICES_OBSERVACAO_26_PARA_37 = (
    0,
    1,
    2,
    3,
    6,
    *range(16, 37),
)


def inicializar_genoma(rng: np.random.Generator) -> dict[str, np.ndarray]:
    genoma: dict[str, np.ndarray] = {}
    pares_dimensoes = {
        "camada1_pesos": (OBSERVACAO_DIM, NEURONIOS_OCULTOS),
        "camada2_pesos": (NEURONIOS_OCULTOS, NEURONIOS_OCULTOS),
        "saida_pesos": (NEURONIOS_OCULTOS, ACOES_DIM),
    }

    for nome, forma in CHAVES_GENOMA.items():
        if nome in pares_dimensoes:
            fan_in, fan_out = pares_dimensoes[nome]
            limite = math.sqrt(6.0 / (fan_in + fan_out))
            genoma[nome] = rng.uniform(-limite, limite, size=forma).astype(np.float32)
        else:
            genoma[nome] = np.zeros(forma, dtype=np.float32)
    return genoma


def validar_genoma(genoma: Mapping[str, np.ndarray]) -> None:
    if set(genoma) != set(CHAVES_GENOMA):
        raise ValueError("O genoma deve conter exatamente os seis tensores da política.")
    for nome, forma in CHAVES_GENOMA.items():
        valor = np.asarray(genoma[nome])
        if valor.shape != forma:
            raise ValueError(
                f"Forma inválida para {nome}: esperado {forma}, recebido {valor.shape}."
            )
        if not np.isfinite(valor).all():
            raise ValueError(f"O tensor {nome} contém valores não finitos.")


def clonar_genoma(genoma: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    validar_genoma(genoma)
    return {nome: np.array(valor, dtype=np.float32, copy=True) for nome, valor in genoma.items()}


def cruzar_genomas(
    pai_a: Mapping[str, np.ndarray],
    pai_b: Mapping[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    validar_genoma(pai_a)
    validar_genoma(pai_b)
    return {
        nome: np.where(rng.random(CHAVES_GENOMA[nome]) < 0.5, pai_a[nome], pai_b[nome])
        .astype(np.float32)
        .copy()
        for nome in CHAVES_GENOMA
    }


def mutar_genoma(
    genoma: Mapping[str, np.ndarray],
    rng: np.random.Generator,
    taxa: float = TAXA_MUTACAO,
    desvio: float = DESVIO_MUTACAO,
) -> dict[str, np.ndarray]:
    if not 0.0 <= taxa <= 1.0:
        raise ValueError("A taxa de mutação deve estar entre 0 e 1.")
    if desvio < 0.0:
        raise ValueError("O desvio da mutação não pode ser negativo.")

    mutado = clonar_genoma(genoma)
    for valor in mutado.values():
        mascara = rng.random(valor.shape) < taxa
        valor[mascara] += rng.normal(0.0, desvio, size=int(mascara.sum())).astype(
            np.float32
        )
    return mutado


def criar_populacao_inicial(
    rng: np.random.Generator,
    genoma_base: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, np.ndarray]]:
    if genoma_base is None:
        return [inicializar_genoma(rng) for _ in range(POPULACAO)]

    validar_genoma(genoma_base)
    return [
        clonar_genoma(genoma_base),
        mutar_genoma(genoma_base, rng),
        mutar_genoma(genoma_base, rng),
    ]


def criar_proxima_geracao(
    avaliados: list[tuple[float, dict[str, np.ndarray]]],
    rng: np.random.Generator,
) -> list[dict[str, np.ndarray]]:
    if len(avaliados) != POPULACAO:
        raise ValueError(f"Cada geração precisa ter exatamente {POPULACAO} agentes.")

    pais = sorted(avaliados, key=lambda candidato: candidato[0], reverse=True)
    melhor = pais[0][1]
    segundo = pais[1][1]
    return [
        clonar_genoma(melhor),
        mutar_genoma(cruzar_genomas(melhor, segundo, rng), rng),
        mutar_genoma(cruzar_genomas(melhor, segundo, rng), rng),
    ]


def calcular_acao(genoma: Mapping[str, np.ndarray], observacao: np.ndarray) -> np.ndarray:
    validar_genoma(genoma)
    entrada = np.asarray(observacao, dtype=np.float32)
    if entrada.shape != (OBSERVACAO_DIM,):
        raise ValueError(
            f"Observação precisa ter {OBSERVACAO_DIM} valores; recebidos {entrada.shape}."
        )

    oculta1 = np.maximum(genoma["camada1_pesos"] @ entrada + genoma["camada1_vies"], 0.0)
    oculta2 = np.maximum(
        genoma["camada2_pesos"] @ oculta1 + genoma["camada2_vies"],
        0.0,
    )
    logits = genoma["saida_pesos"] @ oculta2 + genoma["saida_vies"]
    return (logits > 0.0).astype(np.int8)


def genoma_de_policy(
    state_dict: Mapping[str, torch.Tensor],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    genoma = inicializar_genoma(rng)
    for nome_genoma, nome_policy in PARAMETROS_POLICY.items():
        peso = state_dict.get(nome_policy)
        if peso is None:
            continue
        array = peso.detach().cpu().numpy()

        if nome_genoma == "camada1_pesos" and array.shape[0] == NEURONIOS_OCULTOS:
            if array.shape[1] == OBSERVACAO_DIM:
                genoma[nome_genoma] = array.astype(np.float32, copy=True)
            elif array.shape[1] == 26:
                for indice_antigo, indice_novo in enumerate(
                    INDICES_OBSERVACAO_26_PARA_37
                ):
                    genoma[nome_genoma][:, indice_novo] = array[:, indice_antigo]
        elif nome_genoma in {"saida_pesos", "saida_vies"}:
            linhas = min(array.shape[0], genoma[nome_genoma].shape[0])
            if array.shape[1:] == genoma[nome_genoma].shape[1:]:
                genoma[nome_genoma][:linhas] = array[:linhas]
        elif array.shape == genoma[nome_genoma].shape:
            genoma[nome_genoma] = array.astype(np.float32, copy=True)

    validar_genoma(genoma)
    return genoma


def transferir_genoma_para_ppo(modelo, genoma: Mapping[str, np.ndarray]) -> None:
    validar_genoma(genoma)
    state = modelo.policy.state_dict()
    for nome_genoma, nome_policy in PARAMETROS_POLICY.items():
        atual = state[nome_policy]
        tensor = torch.as_tensor(genoma[nome_genoma], dtype=atual.dtype, device=atual.device)
        if tensor.shape != atual.shape:
            raise ValueError(
                f"Forma incompatível ao transferir {nome_genoma}: "
                f"política {tuple(atual.shape)}, genoma {tuple(tensor.shape)}."
            )
        state[nome_policy] = tensor
    modelo.policy.load_state_dict(state)


def transferir_pesos_ppo(origem, destino) -> list[str]:
    origem_state = origem.policy.state_dict()
    destino_state = destino.policy.state_dict()
    transferidos: list[str] = []

    for nome, peso_destino in destino_state.items():
        peso_origem = origem_state.get(nome)
        if peso_origem is None:
            continue

        if peso_origem.shape == peso_destino.shape:
            destino_state[nome] = peso_origem.detach().clone().to(peso_destino.device)
            transferidos.append(nome)
            continue

        if (
            nome in {
                "mlp_extractor.policy_net.0.weight",
                "mlp_extractor.value_net.0.weight",
            }
            and peso_origem.ndim == 2
            and peso_destino.ndim == 2
            and peso_origem.shape[0] == peso_destino.shape[0]
            and peso_origem.shape[1] == 26
            and peso_destino.shape[1] == OBSERVACAO_DIM
        ):
            transferido = peso_destino.clone()
            for indice_antigo, indice_novo in enumerate(
                INDICES_OBSERVACAO_26_PARA_37
            ):
                transferido[:, indice_novo] = peso_origem[:, indice_antigo].to(
                    peso_destino.device
                )
            destino_state[nome] = transferido
            transferidos.append(f"{nome} (mapeamento da observação antiga)")
            continue

        if (
            nome in {"action_net.weight", "action_net.bias"}
            and peso_origem.ndim == peso_destino.ndim
            and peso_origem.shape[1:] == peso_destino.shape[1:]
            and peso_origem.shape[0] < peso_destino.shape[0]
        ):
            transferido = peso_destino.clone()
            transferido[: peso_origem.shape[0]] = peso_origem.to(peso_destino.device)
            destino_state[nome] = transferido
            transferidos.append(f"{nome} (ações existentes preservadas)")

    destino.policy.load_state_dict(destino_state)
    return transferidos


def salvar_genoma(
    caminho: str,
    genoma: Mapping[str, np.ndarray],
    geracao: int,
    agente: int,
    pontuacao: float,
) -> None:
    validar_genoma(genoma)
    np.savez_compressed(
        caminho,
        **genoma,
        geracao=np.int32(geracao),
        agente=np.int32(agente),
        pontuacao=np.float32(pontuacao),
        observacao_dim=np.int32(OBSERVACAO_DIM),
        acoes_dim=np.int32(ACOES_DIM),
    )


def carregar_genoma(caminho: str) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    with np.load(caminho, allow_pickle=False) as arquivo:
        if int(arquivo["observacao_dim"]) != OBSERVACAO_DIM:
            raise ValueError("O genoma foi treinado com outra dimensão de observação.")
        if int(arquivo["acoes_dim"]) != ACOES_DIM:
            raise ValueError("O genoma foi treinado com outra dimensão de ações.")
        genoma = {nome: np.array(arquivo[nome], copy=True) for nome in CHAVES_GENOMA}
        metadados: dict[str, float | int] = {
            "geracao": int(arquivo["geracao"]),
            "agente": int(arquivo["agente"]),
            "pontuacao": float(arquivo["pontuacao"]),
        }
    validar_genoma(genoma)
    return genoma, metadados
