ID = "etapa_02_sobreviver"
TITULO = "Aprender a sobreviver"
PROXIMA_ETAPA = "dano"
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
    recompensa_sobrevivencia = 0.01 if vida_hornet is not None and vida_hornet > 0 else 0.0
    penalidade_dano = 0.2 * dano_recebido
    penalidade_morte = 2.0 if morreu else 0.0
    recompensa_vitoria = 0.5 if venceu else 0.0
    recompensa = (
        recompensa_sobrevivencia
        - penalidade_dano
        - penalidade_morte
        + recompensa_vitoria
    )
    return recompensa, {
        "sobrevivencia": recompensa_sobrevivencia,
        "dano_recebido": -penalidade_dano,
        "morte": -penalidade_morte,
        "vitoria": recompensa_vitoria,
    }
