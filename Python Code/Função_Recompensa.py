
def inicializar_recompensa():
    return {
        "dano_total": 0,
        "golpes_recebidos": 0,
        "tempo_total": 0,
        "hp_chefe_anterior": None,
        "hp_hornet_anterior": None
    }


def calcular_recompensa(
    estado,
    hp_chefe,
    hp_hornet,
    hp_chefe_max,
    delta_time,
    terminado=False,
    vitoria=False,
    max_time=120,
    lambda_hit=0.2,
    lambda_time=0.1,
    victory_bonus=1.0,
    time_bonus_weight=2.0
):
    # Inicializar HP anterior na primeira execução
    if estado["hp_chefe_anterior"] is None:
        estado["hp_chefe_anterior"] = hp_chefe
        estado["hp_hornet_anterior"] = hp_hornet

    # 1. Dano causado
    dano = max(
        0,
        estado["hp_chefe_anterior"] - hp_chefe
    )

    estado["dano_total"] += dano

    recompensa_dano = (
        2 * dano / hp_chefe_max
    )

    # 2. Golpes recebidos
    golpe = hp_hornet < estado["hp_hornet_anterior"]

    if golpe:
        estado["golpes_recebidos"] += 1

    penalidade_golpe = lambda_hit if golpe else 0

    # 3. Tempo
    estado["tempo_total"] += delta_time

    penalidade_tempo = (
        lambda_time * delta_time / max_time
    )

    # 4. Recompensa durante a luta
    recompensa = (
        recompensa_dano
        - penalidade_golpe
        - penalidade_tempo
    )

    # 5. Bônus de finalização
    bonus_vitoria = 0
    bonus_tempo = 0

    if terminado and vitoria:
        bonus_vitoria = victory_bonus

        bonus_tempo = (
            time_bonus_weight
            * max(
                0,
                1 - estado["tempo_total"] / max_time
            )
        )

    recompensa += bonus_vitoria + bonus_tempo

    # Atualizar HP anterior
    estado["hp_chefe_anterior"] = hp_chefe
    estado["hp_hornet_anterior"] = hp_hornet

    return recompensa, {
        "dano": recompensa_dano,
        "golpe": penalidade_golpe,
        "tempo": penalidade_tempo,
        "vitoria": bonus_vitoria,
        "velocidade": bonus_tempo
    }


def avaliar_combate(
        
    estado,
    derrotou_chefe,
    max_time=120,
    max_hits=10
):
    # 1. Recompensa de tempo
    recompensa_tempo = max(
        0,
        1 - estado["tempo_total"] / max_time
    )

    # 2. Dano causado
    # Pressupõe que o HP máximo do chefe foi registrado externamente.
    recompensa_dano = min(
        1,
        estado["dano_total"] / estado["hp_chefe_max"]
    )

    # 3. Golpes recebidos
    recompensa_golpes = 1 - min(
        1,
        estado["golpes_recebidos"] / max_hits
    )

    # 4. Pontuação final
    pontuacao = (
        2 * recompensa_tempo
        + 2 * recompensa_dano
        + recompensa_golpes
    ) / 5

    return {
        "pontuacao": pontuacao,
        "vitoria": derrotou_chefe,
        "tempo": estado["tempo_total"],
        "dano": estado["dano_total"],
        "golpes": estado["golpes_recebidos"]
    }