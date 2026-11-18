ID = "etapa_04_vencer"
TITULO = "Aprender a vencer"
PROXIMA_ETAPA = None
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
    recompensa_dano = 10.0 * dano_causado / HP_CHEFE_MAX
    penalidade_dano_recebido = 0.2 * dano_recebido
    penalidade_morte = 2.0 if morreu else 0.0
    recompensa_vitoria = 10.0 if venceu else 0.0
    bonus_vida_final = (
        float(vida_hornet) / HP_HORNET_MAX
        if venceu and vida_hornet is not None
        else 0.0
    )
    recompensa = (
        recompensa_dano
        - penalidade_dano_recebido
        - penalidade_morte
        + recompensa_vitoria
        + bonus_vida_final
    )
    return recompensa, {
        "dano_chefe": recompensa_dano,
        "dano_recebido": -penalidade_dano_recebido,
        "morte": -penalidade_morte,
        "vitoria": recompensa_vitoria,
        "vida_final": bonus_vida_final,
    }
