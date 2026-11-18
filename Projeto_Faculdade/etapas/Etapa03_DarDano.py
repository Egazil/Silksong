ID = "etapa_03_dar_dano"
TITULO = "Aprender a causar dano"
PROXIMA_ETAPA = "vencer"
HP_CHEFE_MAX = 120
HP_HORNET_MAX = 5
MAX_EPISODE_STEPS = 5000
LIMIAR_HP_CHEFE_EPISODIO = 20
PENALIDADE_TEMPO = 0.01
RECOMPENSA_VITORIA = 30.0
MULTIPLICADOR_VITORIA_VIDA = 10.0


def calcular_recompensa(
    dano_causado,
    dano_recebido,
    cura_realizada,
    morreu,
    venceu,
    vida_hornet,
    recompensa_proximidade=0.0,
    penalidade_tempo=None,
    penalidade_parede=0.0,
):
    custo_tempo = PENALIDADE_TEMPO if penalidade_tempo is None else penalidade_tempo
    multiplicador_vida = max(
        0.1,
        max(0.0, float(vida_hornet or 0.0)) / HP_HORNET_MAX,
    )
    recompensa_dano = (
        (dano_causado / HP_CHEFE_MAX) * 20.0 * multiplicador_vida
    )
    penalidade_dano_recebido = float(dano_recebido)
    recompensa_vitoria = (
        RECOMPENSA_VITORIA
        + MULTIPLICADOR_VITORIA_VIDA
        * max(0.0, float(vida_hornet or 0.0))
        / HP_HORNET_MAX
        if venceu
        else 0.0
    )
    recompensa = (
        recompensa_dano
        - penalidade_dano_recebido
        - custo_tempo
        + recompensa_proximidade
        - penalidade_parede
        + recompensa_vitoria
    )
    return recompensa, {
        "dano_chefe": recompensa_dano,
        "dano_recebido": -penalidade_dano_recebido,
        "tempo": -custo_tempo,
        "proximidade": recompensa_proximidade,
        "parede": -penalidade_parede,
        "vitoria": recompensa_vitoria,
    }
