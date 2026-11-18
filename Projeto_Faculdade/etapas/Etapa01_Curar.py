ID = "etapa_01_curar"
TITULO = "Aprender a curar"
PROXIMA_ETAPA = "sobreviver"
HP_CHEFE_MAX = 120
HP_HORNET_MAX = 5
MAX_EPISODE_STEPS = 1200


def calcular_recompensa(
    dano_causado,
    dano_recebido,
    cura_realizada,
    morreu,
    venceu,
    vida_hornet,
):
    recompensa_cura = 2.0 * cura_realizada
    penalidade_dano = 0.1 * dano_recebido
    penalidade_morte = 2.0 if morreu else 0.0
    recompensa_vitoria = 0.0
    recompensa = (
        recompensa_cura
        - penalidade_dano
        - penalidade_morte
        + recompensa_vitoria
    )
    return recompensa, {
        "cura": recompensa_cura,
        "dano_recebido": -penalidade_dano,
        "morte": -penalidade_morte,
        "vitoria": recompensa_vitoria,
    }
