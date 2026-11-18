import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from ambiente_silksong import SilksongEnv
from etapas import ORDEM_ETAPAS, obter_etapa
from genetico import (
    ACOES_DIM,
    GERACOES,
    OBSERVACAO_DIM,
    PARAMETROS_POLICY,
    POPULACAO,
    INDICES_OBSERVACAO_26_PARA_37,
    calcular_acao,
    carregar_genoma,
    criar_populacao_inicial,
    criar_proxima_geracao,
    inicializar_genoma,
    mutar_genoma,
    salvar_genoma,
    transferir_genoma_para_ppo,
    transferir_pesos_ppo,
)
from treinar_genetico import gerar_grafico_evolucao
from treinar_silksong import MonitoramentoCompletoCallback


class EtapasCurriculoTests(unittest.TestCase):
    def test_ordem_do_curriculo(self):
        self.assertEqual(ORDEM_ETAPAS, ("curar", "sobreviver", "dano", "vencer"))

    def test_etapa_curar_recompensa_recuperacao_real(self):
        etapa = obter_etapa("curar")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=0,
            dano_recebido=0,
            cura_realizada=2,
            morreu=False,
            venceu=False,
            vida_hornet=5,
        )
        self.assertEqual(recompensa, 4.0)
        self.assertEqual(componentes["cura"], 4.0)

    def test_etapa_sobreviver_recompensa_permanecer_vivo(self):
        etapa = obter_etapa("sobreviver")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=0,
            dano_recebido=0,
            cura_realizada=0,
            morreu=False,
            venceu=False,
            vida_hornet=3,
        )
        self.assertEqual(recompensa, 0.01)
        self.assertEqual(componentes["sobrevivencia"], 0.01)

    def test_etapa_dano_reforca_reducao_de_hp_do_chefe(self):
        etapa = obter_etapa("dano")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=12,
            dano_recebido=0,
            cura_realizada=0,
            morreu=False,
            venceu=False,
            vida_hornet=5,
        )
        self.assertEqual(recompensa, 1.99)
        self.assertEqual(componentes["dano_chefe"], 2.0)
        self.assertEqual(componentes["tempo"], -0.01)

    def test_etapa_dano_pune_dano_recebido_sem_pontuar_derrota(self):
        etapa = obter_etapa("dano")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=0,
            dano_recebido=1,
            cura_realizada=0,
            morreu=True,
            venceu=False,
            vida_hornet=4,
        )
        self.assertEqual(recompensa, -1.01)
        self.assertEqual(componentes["dano_recebido"], -1.0)
        self.assertNotIn("morte", componentes)

    def test_etapa_dano_pode_remover_custo_de_tempo_para_dano_parado(self):
        etapa = obter_etapa("dano")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=12,
            dano_recebido=0,
            cura_realizada=0,
            morreu=False,
            venceu=False,
            vida_hornet=5,
            penalidade_tempo=0.0,
        )
        self.assertEqual(recompensa, 2.0)
        self.assertEqual(componentes["tempo"], 0.0)

    def test_etapa_dano_pune_cruzamento_de_parede_sem_custo_de_chao(self):
        etapa = obter_etapa("dano")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=0,
            dano_recebido=0,
            cura_realizada=0,
            morreu=False,
            venceu=False,
            vida_hornet=5,
            penalidade_tempo=0.0,
            penalidade_parede=1.0,
        )
        self.assertEqual(recompensa, -1.0)
        self.assertEqual(componentes["parede"], -1.0)

    def test_etapa_dano_bonifica_vitoria_e_progresso_de_proximidade(self):
        etapa = obter_etapa("dano")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=0,
            dano_recebido=0,
            cura_realizada=0,
            morreu=False,
            venceu=True,
            vida_hornet=5,
            recompensa_proximidade=0.1,
        )
        self.assertAlmostEqual(recompensa, 40.09)
        self.assertEqual(componentes["vitoria"], 40.0)
        self.assertEqual(componentes["proximidade"], 0.1)

    def test_etapa_vencer_premia_vitoria(self):
        etapa = obter_etapa("vencer")
        recompensa, componentes = etapa.calcular_recompensa(
            dano_causado=0,
            dano_recebido=0,
            cura_realizada=0,
            morreu=False,
            venceu=True,
            vida_hornet=5,
        )
        self.assertEqual(recompensa, 11.0)
        self.assertEqual(componentes["vitoria"], 10.0)

    def test_rejeita_etapa_desconhecida(self):
        with self.assertRaises(ValueError):
            obter_etapa("pular")

    def test_observacao_inclui_velocidades_estado_e_limites(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.etapa = obter_etapa("dano")
        env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVACAO_DIM,), dtype=np.float32
        )
        dados = {
            "vida_hornet": 4,
            "vida_hornet_valida": True,
            "pode_focar": True,
            "hornet_x": 50,
            "hornet_y": 20,
            "hornet_vx": 1.5,
            "hornet_vy": -0.5,
            "vida_mae": 100,
            "vida_mae_valida": True,
            "mae_x": 57.5,
            "mae_y": 23,
            "mae_vx": 0,
            "mae_vy": 0,
            "mae_estado_codigo": 1,
            "arena_min_x": 40,
            "arena_max_x": 75,
            "arena_min_y": 12,
            "arena_max_y": 35,
            "alvos": [
                {"nome": "Mossbone Mother", "posX": 57.5, "posY": 23, "distancia": 7.5}
            ],
        }

        obs, _, _ = env._construir_observacao(dados)

        self.assertEqual(obs.shape, (37,))
        np.testing.assert_array_equal(
            obs[:16],
            np.array(
                [4, 1, 50, 20, 1.5, -0.5, 100, 57.5, 23, 0, 0, 1, 40, 75, 12, 35],
                dtype=np.float32,
            ),
        )
        self.assertEqual(obs[16], 7.5)
        self.assertEqual(obs[17], 3.0)
        self.assertEqual(obs[18], 7.5)
        self.assertEqual(obs[21], 999.0)

    def test_recompensa_proximidade_premia_aproximar_e_pune_afastar(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.etapa = obter_etapa("dano")
        env.recompensa_estado = {"distancia_chefe_anterior": None}
        dados = {
            "hornet_detectada": True,
            "chefe_detectada": True,
            "vida_hornet_valida": True,
            "vida_mae_valida": True,
            "hornet_x": 10.0,
            "hornet_y": 10.0,
            "mae_x": 20.0,
            "mae_y": 10.0,
            "arena_min_x": 0.0,
            "arena_max_x": 30.0,
            "arena_min_y": 0.0,
            "arena_max_y": 40.0,
        }

        self.assertEqual(env._calcular_recompensa_proximidade(dados), 0.0)
        dados["hornet_x"] = 11.0
        self.assertAlmostEqual(env._calcular_recompensa_proximidade(dados), 0.005)
        dados["hornet_x"] = 10.0
        self.assertAlmostEqual(env._calcular_recompensa_proximidade(dados), -0.005)

    def test_penalidade_parede_e_aplicada_so_ao_cruzar_as_laterais(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.etapa = obter_etapa("dano")
        env.penalidade_parede = 1.0
        env.recompensa_estado = {"posicao_x_hornet_anterior": None}
        dados = {
            "hornet_detectada": True,
            "hornet_x": 10.0,
            "hornet_y": -100.0,
            "arena_min_x": 0.0,
            "arena_max_x": 20.0,
        }

        self.assertEqual(env._calcular_penalidade_parede(dados), 0.0)
        dados["hornet_x"] = -0.1
        self.assertEqual(env._calcular_penalidade_parede(dados), 1.0)
        dados["hornet_x"] = -1.0
        self.assertEqual(env._calcular_penalidade_parede(dados), 0.0)
        dados["hornet_x"] = 0.1
        self.assertEqual(env._calcular_penalidade_parede(dados), 1.0)
        dados["hornet_x"] = 20.1
        self.assertEqual(env._calcular_penalidade_parede(dados), 1.0)

    def test_monitor_encerra_apos_limite_de_episodios(self):
        callback = MonitoramentoCompletoCallback(
            etapa_id="etapa_03_dar_dano",
            pasta_videos="",
            caminho_grafico="",
            habilitar_video=False,
            limite_episodios=2,
        )
        callback.update_locals({"infos": [{"episode": {"r": 1.0, "l": 10}}]})
        with patch("builtins.print"):
            self.assertTrue(callback._on_step())

        callback.update_locals({"infos": [{"episode": {"r": 2.0, "l": 10}}]})
        with patch("builtins.print"):
            self.assertFalse(callback._on_step())

    def test_limite_de_120_segundos_do_jogo_encerra_com_derrota(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.max_episode_seconds = 120.0
        env.max_episode_steps = 1200
        env._tempo_jogo_inicio = 10.0
        env.penalidade_tempo = None
        env.penalidade_parede = None
        env.episode_steps = 0
        env.etapa = obter_etapa("dano")
        env._motivo_ultimo_termino = None
        env.recompensa_estado = {
            "hp_chefe_anterior": 120.0,
            "hp_hornet_anterior": 5.0,
            "dano_total": 0,
            "golpes_recebidos": 0,
            "recompensa_acumulada_episodio": 0.0,
        }
        dados = {
            "tempo_jogo": 130.0,
            "vida_hornet": 5,
            "vida_hornet_valida": True,
            "vida_mae": 120,
            "vida_mae_valida": True,
            "hornet_detectada": True,
            "morte_real_confirmada": False,
            "vitoria_real_confirmada": False,
        }
        env._aplicar_inputs_teclado = lambda _: None
        env._ler_dados_tcp = lambda: dados
        env._construir_observacao = lambda _: (
            np.zeros(OBSERVACAO_DIM, dtype=np.float32),
            5.0,
            120.0,
        )
        env._liberar_todas_teclas = lambda: None
        env._registrar_telemetria = lambda *_: None

        _, recompensa, terminado, truncado, info = env.step(np.zeros(7, dtype=np.int8))

        self.assertTrue(terminado)
        self.assertFalse(truncado)
        self.assertEqual(recompensa, -0.01)
        self.assertTrue(info["derrota_por_tempo"])
        self.assertEqual(info["tempo_decorrido_jogo"], 120.0)
        self.assertEqual(env._motivo_ultimo_termino, "limite_tempo")

        env.max_episode_seconds = None
        env._tempo_jogo_inicio = None
        env.penalidade_tempo = None
        env.penalidade_parede = None
        env.episode_steps = env.max_episode_steps - 1
        env._motivo_ultimo_termino = None
        env.recompensa_estado = {
            "hp_chefe_anterior": 120.0,
            "hp_hornet_anterior": 5.0,
            "dano_total": 0,
            "golpes_recebidos": 0,
            "recompensa_acumulada_episodio": 0.0,
        }

        _, _, terminado, truncado, _ = env.step(np.zeros(7, dtype=np.int8))

        self.assertFalse(terminado)
        self.assertTrue(truncado)
        self.assertEqual(env._motivo_ultimo_termino, "limite_steps")

    def test_vitoria_encerra_partida_com_bonus_e_motivo_de_reset(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.max_episode_seconds = None
        env._tempo_jogo_inicio = None
        env.penalidade_tempo = None
        env.penalidade_parede = None
        env.max_episode_steps = 1200
        env.episode_steps = 0
        env.etapa = obter_etapa("dano")
        env._motivo_ultimo_termino = None
        env.recompensa_estado = {
            "hp_chefe_anterior": 1.0,
            "hp_hornet_anterior": 5.0,
            "distancia_chefe_anterior": 10.0,
            "dano_total": 0,
            "golpes_recebidos": 0,
            "recompensa_acumulada_episodio": 0.0,
        }
        dados = {
            "vida_hornet": 5,
            "vida_hornet_valida": True,
            "vida_mae": 0,
            "vida_mae_valida": True,
            "hornet_detectada": True,
            "chefe_detectada": True,
            "morte_real_confirmada": False,
            "vitoria_real_confirmada": False,
            "hornet_x": 10.0,
            "hornet_y": 10.0,
            "mae_x": 20.0,
            "mae_y": 10.0,
            "arena_min_x": 0.0,
            "arena_max_x": 30.0,
            "arena_min_y": 0.0,
            "arena_max_y": 40.0,
        }
        env._aplicar_inputs_teclado = lambda _: None
        env._ler_dados_tcp = lambda: dados
        env._construir_observacao = lambda _: (
            np.zeros(OBSERVACAO_DIM, dtype=np.float32),
            5.0,
            0.0,
        )
        env._liberar_todas_teclas = lambda: None
        env._registrar_telemetria = lambda *_: None

        with patch("builtins.print"):
            _, _, terminado, truncado, info = env.step(np.zeros(7, dtype=np.int8))

        self.assertTrue(terminado)
        self.assertFalse(truncado)
        self.assertEqual(env._motivo_ultimo_termino, "vitoria")
        self.assertEqual(info["vitoria"], 40.0)
        self.assertEqual(info["episode"]["l"], 1)

    def test_vitoria_confirmada_com_hp_indisponivel_encerra_e_contabiliza_hp_final(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.max_episode_seconds = None
        env._tempo_jogo_inicio = None
        env.penalidade_tempo = None
        env.penalidade_parede = None
        env.max_episode_steps = 5000
        env.episode_steps = 10
        env.etapa = obter_etapa("dano")
        env._motivo_ultimo_termino = None
        env.recompensa_estado = {
            "hp_chefe_anterior": 5.0,
            "hp_hornet_anterior": 5.0,
            "distancia_chefe_anterior": None,
            "posicao_x_hornet_anterior": None,
            "dano_total": 0,
            "golpes_recebidos": 0,
            "recompensa_acumulada_episodio": 0.0,
        }
        dados = {
            "vida_hornet": 5,
            "vida_hornet_valida": True,
            "vida_mae": -1,
            "vida_mae_valida": False,
            "hornet_detectada": True,
            "vitoria_real_confirmada": True,
            "morte_real_confirmada": False,
            "hornet_x": 10.0,
            "hornet_y": 10.0,
            "mae_x": 20.0,
            "mae_y": 10.0,
            "arena_min_x": 0.0,
            "arena_max_x": 30.0,
            "arena_min_y": 0.0,
            "arena_max_y": 40.0,
        }
        env._aplicar_inputs_teclado = lambda _: None
        env._ler_dados_tcp = lambda: dados
        env._construir_observacao = lambda _: (
            np.zeros(OBSERVACAO_DIM, dtype=np.float32),
            5.0,
            -1.0,
        )
        env._liberar_todas_teclas = lambda: None
        env._registrar_telemetria = lambda *_: None

        with patch("builtins.print"):
            _, _, terminado, truncado, info = env.step(np.zeros(7, dtype=np.int8))

        self.assertTrue(terminado)
        self.assertFalse(truncado)
        self.assertEqual(env._motivo_ultimo_termino, "vitoria")
        self.assertEqual(info["dano_causado"], 5.0)
        self.assertEqual(info["vitoria"], 40.0)

    def test_etapa_dano_encerra_e_solicita_reset_ao_chegar_a_20_hp(self):
        env = SilksongEnv.__new__(SilksongEnv)
        env.max_episode_seconds = None
        env._tempo_jogo_inicio = None
        env.penalidade_tempo = None
        env.penalidade_parede = None
        env.max_episode_steps = 5000
        env.episode_steps = 15
        env.etapa = obter_etapa("dano")
        env._motivo_ultimo_termino = None
        env.recompensa_estado = {
            "hp_chefe_anterior": 25.0,
            "hp_hornet_anterior": 5.0,
            "distancia_chefe_anterior": None,
            "posicao_x_hornet_anterior": None,
            "dano_total": 0,
            "golpes_recebidos": 0,
            "recompensa_acumulada_episodio": 0.0,
        }
        dados = {
            "vida_hornet": 5,
            "vida_hornet_valida": True,
            "vida_mae": 20,
            "vida_mae_valida": True,
            "hornet_detectada": True,
            "vitoria_real_confirmada": False,
            "morte_real_confirmada": False,
            "hornet_x": 10.0,
            "hornet_y": 10.0,
            "mae_x": 20.0,
            "mae_y": 10.0,
            "arena_min_x": 0.0,
            "arena_max_x": 30.0,
            "arena_min_y": 0.0,
            "arena_max_y": 40.0,
        }
        env._aplicar_inputs_teclado = lambda _: None
        env._ler_dados_tcp = lambda: dados
        env._construir_observacao = lambda _: (
            np.zeros(OBSERVACAO_DIM, dtype=np.float32),
            5.0,
            20.0,
        )
        env._liberar_todas_teclas = lambda: None
        env._registrar_telemetria = lambda *_: None

        with patch("builtins.print"):
            _, _, terminado, truncado, info = env.step(np.zeros(7, dtype=np.int8))

        self.assertTrue(terminado)
        self.assertFalse(truncado)
        self.assertTrue(info["limite_hp_chefe"])
        self.assertFalse(info["vitoria"])
        self.assertEqual(info["dano_causado"], 5.0)
        self.assertEqual(env._motivo_ultimo_termino, "limite_hp_chefe")

    def test_monitor_imprime_pontuacao_de_cada_episodio_e_salva_png(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from testar_treino_dano import (
            gerar_grafico_pontuacoes,
            ler_pontuacoes_monitor,
        )

        with tempfile.TemporaryDirectory() as pasta:
            caminho_grafico = Path(pasta) / "pontuacoes.png"
            callback = MonitoramentoCompletoCallback(
                etapa_id="etapa_03_dar_dano",
                pasta_videos=pasta,
                caminho_grafico=str(caminho_grafico),
                habilitar_video=False,
            )
            callback.update_locals(
                {"infos": [{"episode": {"r": 12.5, "l": 8}}]}
            )
            with patch("builtins.print") as imprimir:
                callback._on_step()
            self.assertTrue(
                any("Pontuação: 12.50" in str(chamada) for chamada in imprimir.call_args_list)
            )

            with patch("builtins.print"):
                self.assertTrue(
                    gerar_grafico_pontuacoes(
                        callback.recompensas_episodios,
                        caminho_grafico,
                    )
                )
            self.assertTrue(caminho_grafico.is_file())
            self.assertGreater(caminho_grafico.stat().st_size, 0)
            self.assertEqual(caminho_grafico.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            caminho_monitor = Path(pasta) / "monitor.csv"
            caminho_monitor.write_text(
                '#{"t_start": 1.0}\nr,l,t\n18.694306,1211,231.5\n',
                encoding="utf-8",
            )
            self.assertEqual(
                ler_pontuacoes_monitor(caminho_monitor),
                [18.694306],
            )

    def test_primeiro_inicio_preserva_estado_atual_quando_configurado(self):
        class ConexaoFalsa:
            def gettimeout(self):
                return None

            def settimeout(self, _timeout):
                pass

        env = SilksongEnv.__new__(SilksongEnv)
        env.etapa = obter_etapa("dano")
        env.max_episode_seconds = 120.0
        env.max_episode_steps = 1200
        env.episode_steps = 0
        env._tempo_jogo_inicio = None
        env._sessao_iniciada = False
        env._reset_id = 0
        env._motivo_ultimo_termino = None
        env.preservar_estado_inicial = True
        env.recompensa_estado = {
            "hp_chefe_anterior": None,
            "hp_hornet_anterior": None,
            "dano_total": 0,
            "golpes_recebidos": 0,
            "recompensa_acumulada_episodio": 0.0,
        }
        env.conn = ConexaoFalsa()
        env.comandos = []
        env._enviar_comando_tcp = env.comandos.append
        env.resposta_reset = {
            "reset_id": 1,
            "reset_completo": True,
            "vida_hornet_valida": True,
            "vida_hornet": 2,
            "vida_mae_valida": True,
            "vida_mae": 77,
            "tempo_jogo": 42.5,
        }
        env._ler_proxima_mensagem_tcp = lambda: env.resposta_reset
        env._liberar_todas_teclas = lambda: None
        env._construir_observacao = lambda _dados: (
            np.zeros(OBSERVACAO_DIM, dtype=np.float32),
            2.0,
            77.0,
        )
        env._registrar_telemetria = lambda *_args: None

        env.reset()

        self.assertEqual(env.comandos, ["START:1:etapa_03_dar_dano"])
        self.assertEqual(env.recompensa_estado["hp_hornet_anterior"], 2)
        self.assertEqual(env.recompensa_estado["hp_chefe_anterior"], 77.0)
        self.assertEqual(env._tempo_jogo_inicio, 42.5)

        env._motivo_ultimo_termino = "limite_steps"
        env.resposta_reset = {
            "reset_id": 2,
            "reset_completo": True,
            "vida_hornet_valida": True,
            "vida_hornet": 5,
            "vida_mae_valida": True,
            "vida_mae": 120,
            "tempo_jogo": 60.0,
        }
        env.reset()

        self.assertEqual(
            env.comandos,
            [
                "START:1:etapa_03_dar_dano",
                "RESET:2:LIMITE_STEPS:etapa_03_dar_dano",
            ],
        )
        env._motivo_ultimo_termino = "limite_hp_chefe"
        env.resposta_reset = {
            "reset_id": 3,
            "reset_completo": True,
            "vida_hornet_valida": True,
            "vida_hornet": 5,
            "vida_mae_valida": True,
            "vida_mae": 120,
            "tempo_jogo": 75.0,
        }
        env.reset()
        self.assertEqual(
            env.comandos[-1],
            "RESET:3:LIMITE_HP_CHEFE:etapa_03_dar_dano",
        )
        env._motivo_ultimo_termino = "vitoria"
        env.resposta_reset = {
            "reset_id": 4,
            "reset_completo": True,
            "vida_hornet_valida": True,
            "vida_hornet": 5,
            "vida_mae_valida": True,
            "vida_mae": 120,
            "tempo_jogo": 75.0,
        }
        env.reset()
        self.assertEqual(
            env.comandos[-1],
            "RESET:4:VITORIA:etapa_03_dar_dano",
        )

    def test_genoma_produz_acao_multi_binary(self):
        genoma = inicializar_genoma(np.random.default_rng(42))
        acao = calcular_acao(genoma, np.zeros(OBSERVACAO_DIM, dtype=np.float32))
        self.assertEqual(acao.shape, (ACOES_DIM,))
        self.assertTrue(np.isin(acao, (0, 1)).all())

    def test_operadores_geneticos_preservam_populacao_e_formas(self):
        rng = np.random.default_rng(12)
        populacao = criar_populacao_inicial(rng)
        avaliados = [(float(indice), genoma) for indice, genoma in enumerate(populacao)]
        proxima = criar_proxima_geracao(avaliados, rng)

        self.assertEqual(GERACOES, 5)
        self.assertEqual(len(populacao), POPULACAO)
        self.assertEqual(len(proxima), POPULACAO)
        for genoma in proxima:
            acao = calcular_acao(
                genoma, np.zeros(OBSERVACAO_DIM, dtype=np.float32)
            )
            self.assertEqual(acao.shape, (ACOES_DIM,))

    def test_salvar_e_carregar_genoma(self):
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(7)
        genoma = mutar_genoma(inicializar_genoma(rng), rng)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = str(Path(pasta) / "genoma.npz")
            salvar_genoma(caminho, genoma, geracao=3, agente=2, pontuacao=1.25)
            carregado, metadados = carregar_genoma(caminho)

        self.assertEqual(metadados, {"geracao": 3, "agente": 2, "pontuacao": 1.25})
        for nome in genoma:
            np.testing.assert_array_equal(genoma[nome], carregado[nome])

    def test_migracao_preserva_pesos_antigos_e_inicializa_novos_campos(self):
        class PolicyFalsa:
            def __init__(self, state):
                self.state = state

            def state_dict(self):
                return dict(self.state)

            def load_state_dict(self, state):
                self.state = dict(state)

        source_policy = PolicyFalsa(
            {
                "mlp_extractor.policy_net.0.weight": torch.arange(
                    64 * 26, dtype=torch.float32
                ).reshape(64, 26),
                "mlp_extractor.value_net.0.weight": torch.arange(
                    64 * 26, dtype=torch.float32
                ).reshape(64, 26),
                "action_net.weight": torch.arange(
                    6 * 64, dtype=torch.float32
                ).reshape(6, 64),
                "action_net.bias": torch.arange(6, dtype=torch.float32),
            }
        )
        destination_policy = PolicyFalsa(
            {
                "mlp_extractor.policy_net.0.weight": torch.ones(64, 37),
                "mlp_extractor.value_net.0.weight": torch.ones(64, 37),
                "action_net.weight": torch.ones(7, 64),
                "action_net.bias": torch.ones(7),
            }
        )

        transferidos = transferir_pesos_ppo(
            type("Modelo", (), {"policy": source_policy})(),
            type("Modelo", (), {"policy": destination_policy})(),
        )

        self.assertEqual(len(transferidos), 4)
        policy_weight = destination_policy.state[
            "mlp_extractor.policy_net.0.weight"
        ]
        source_weight = source_policy.state["mlp_extractor.policy_net.0.weight"]
        for antigo, novo in enumerate(INDICES_OBSERVACAO_26_PARA_37):
            torch.testing.assert_close(policy_weight[:, novo], source_weight[:, antigo])
        indices_novos = set(range(37)) - set(INDICES_OBSERVACAO_26_PARA_37)
        for indice in indices_novos:
            torch.testing.assert_close(
                policy_weight[:, indice], torch.ones(64, dtype=torch.float32)
            )
        torch.testing.assert_close(
            destination_policy.state["action_net.weight"][:6],
            source_policy.state["action_net.weight"],
        )
        torch.testing.assert_close(
            destination_policy.state["action_net.weight"][6],
            torch.ones(64, dtype=torch.float32),
        )

    def test_transferencia_genetica_copia_pesos_da_politica(self):
        class PolicyFalsa:
            def __init__(self, state):
                self.state = state

            def state_dict(self):
                return dict(self.state)

            def load_state_dict(self, state):
                self.state = dict(state)

        genoma = inicializar_genoma(np.random.default_rng(31))
        estado = {
            nome_policy: torch.zeros_like(torch.from_numpy(genoma[nome_genoma]))
            for nome_genoma, nome_policy in PARAMETROS_POLICY.items()
        }
        policy = PolicyFalsa(estado)

        transferir_genoma_para_ppo(type("Modelo", (), {"policy": policy})(), genoma)

        for nome_genoma, nome_policy in PARAMETROS_POLICY.items():
            torch.testing.assert_close(
                policy.state[nome_policy], torch.from_numpy(genoma[nome_genoma])
            )

    def test_grafico_genetico_e_salvo_com_metricas_por_geracao(self):
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "evolucao_genetica.png"
            gerar_grafico_evolucao([[1.0, 2.0, 3.0], [2.0, 4.0, 3.0]], caminho)

            self.assertTrue(caminho.is_file())
            self.assertGreater(caminho.stat().st_size, 0)
            self.assertEqual(caminho.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
