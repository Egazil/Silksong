import json
import math
import select
import socket
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from etapas import obter_etapa

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.0
except ImportError:
    pydirectinput = None

# Mapeamento dos botões (Ações MultiBinary de tamanho 7)
# 0: Esquerda, 1: Direita, 2: Cima, 3: Baixo, 4: Pular, 5: Atacar, 6: Curar
MAPA_TECLAS = {
    0: "left",   # Esquerda
    1: "right",  # Direita
    2: "up",     # Cima (Olhar/Ataque para cima)
    3: "down",   # Baixo (Olhar/Pogo para baixo)
    4: "z",      # Pulo
    5: "x",      # Ataque
    6: "a",      # Foco/curar (configure esta tecla no jogo)
}

NOMES_ACOES = [
    "ESQUERDA",
    "DIREITA",
    "CIMA",
    "BAIXO",
    "PULAR",
    "ATACAR",
    "CURAR",
]

MAX_ALVOS = 7
VALOR_PADDING_DISTANCIA = 999.0
ESCALA_RECOMPENSA_PROXIMIDADE = 0.25


def inicializar_recompensa():
    return {
        "dano_total": 0,
        "golpes_recebidos": 0,
        "hp_chefe_anterior": None,
        "hp_hornet_anterior": None,
        "distancia_chefe_anterior": None,
        "posicao_x_hornet_anterior": None,
        "recompensa_acumulada_episodio": 0.0,
    }


class SilksongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        etapa="curar",
        host="127.0.0.1",
        port=5001,
        max_episode_steps=None,
        max_episode_seconds=None,
        arquivo_log=None,
        preservar_estado_inicial=False,
        penalidade_tempo=None,
        penalidade_parede=None,
    ):
        super(SilksongEnv, self).__init__()
        self.etapa = obter_etapa(etapa)
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else self.etapa.MAX_EPISODE_STEPS
        )
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps precisa ser maior que zero.")
        if max_episode_seconds is not None and max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds precisa ser maior que zero.")
        if penalidade_tempo is not None and penalidade_tempo < 0:
            raise ValueError("penalidade_tempo não pode ser negativa.")
        if penalidade_parede is not None and penalidade_parede < 0:
            raise ValueError("penalidade_parede não pode ser negativa.")
        self.penalidade_tempo = penalidade_tempo
        self.penalidade_parede = penalidade_parede
        self.max_episode_seconds = max_episode_seconds
        self._tempo_jogo_inicio = None
        self.preservar_estado_inicial = preservar_estado_inicial

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)

        print(
            f"Aguardando conexão do mod C# no IP {host}:{port} (Abra o jogo e entre na arena)..."
        )
        self.conn, self.addr = self.server.accept()
        self.conn.setblocking(True)
        print(f"Mod C# conectado com sucesso via {self.addr}!")

        self.buffer = ""
        self.recompensa_estado = inicializar_recompensa()
        self.episode_steps = 0
        self._sessao_iniciada = False
        self._reset_id = 0
        self._motivo_ultimo_termino = None
        self.teclas_pressionadas = {i: False for i in range(7)}
        self.arquivo_log = arquivo_log or "auditoria_telemetria.jsonl"
        print(
            f"[Currículo] Etapa ativa: {self.etapa.ID} - {self.etapa.TITULO} "
            f"(limite: {self.max_episode_steps} steps)"
        )
        print(f"[Telemetria] Registrando dados e inputs em: '{self.arquivo_log}'")

        # 7 ações independentes: [Esquerda, Direita, Cima, Baixo, Pular, Atacar, Curar]
        self.action_space = spaces.MultiBinary(7)

        # 16 dados de personagem, chefe, estado e arena + 21 dados de alvos.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )

    def _aplicar_inputs_teclado(self, action):
        """
        Executa os inputs reais do teclado usando pydirectinput.
        Permite segurar botões de pulo, andar e direcionar ataques.
        """
        if pydirectinput is None:
            return

        for idx, valor_acao in enumerate(action):
            tecla = MAPA_TECLAS.get(idx)
            if not tecla:
                continue

            deve_pressionar = bool(valor_acao == 1)
            esta_pressionada = self.teclas_pressionadas.get(idx, False)

            if deve_pressionar and not esta_pressionada:
                pydirectinput.keyDown(tecla)
                self.teclas_pressionadas[idx] = True
            elif not deve_pressionar and esta_pressionada:
                pydirectinput.keyUp(tecla)
                self.teclas_pressionadas[idx] = False

    def _liberar_todas_teclas(self):
        """Solta todas as teclas para evitar que continuem pressionadas após o reset/fim."""
        if pydirectinput is None:
            return

        for idx, tecla in MAPA_TECLAS.items():
            try:
                pydirectinput.keyUp(tecla)
            except Exception:
                pass
            self.teclas_pressionadas[idx] = False

    def _enviar_comando_tcp(self, comando: str):
        payload = f"{comando}\n".encode("utf-8")
        self.conn.sendall(payload)

    def _ler_dados_tcp(self):
        """
        Drena mensagens já recebidas e pendentes no socket, retornando o estado mais recente.
        """
        dados = self._ler_proxima_mensagem_tcp()

        while select.select([self.conn], [], [], 0)[0]:
            chunk = self.conn.recv(65536).decode("utf-8", errors="ignore")
            if not chunk:
                raise ConnectionError("Conexão perdida com o mod C# do Silksong.")
            self.buffer += chunk

        while "\n" in self.buffer:
            linha, self.buffer = self.buffer.split("\n", 1)
            try:
                mensagem = json.loads(linha.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(mensagem, dict):
                dados = mensagem
        return dados

    def _ler_proxima_mensagem_tcp(self):
        while True:
            while "\n" in self.buffer:
                linha, self.buffer = self.buffer.split("\n", 1)
                try:
                    mensagem = json.loads(linha.strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(mensagem, dict):
                    return mensagem

            chunk = self.conn.recv(4096).decode("utf-8", errors="ignore")
            if not chunk:
                raise ConnectionError("Conexão perdida com o mod C# do Silksong.")
            self.buffer += chunk

    def _construir_observacao(self, dados):
        """
        Gera o vetor de 37 valores com padding para alvos ausentes.
        """
        vida_h = float(dados.get("vida_hornet", 0))
        if not dados.get("vida_hornet_valida", vida_h >= 0):
            vida_h = -1.0
        pode_focar = float(1.0 if dados.get("pode_focar", False) else 0.0)
        hornet_x = float(dados.get("hornet_x", 0.0))
        hornet_y = float(dados.get("hornet_y", 0.0))
        vida_mae = float(dados.get("vida_mae", -1))
        if not dados.get("vida_mae_valida", vida_mae >= 0) or vida_mae < 0:
            vida_mae = -1.0

        dados_base = [
            vida_h,
            pode_focar,
            float(dados.get("hornet_x", 0.0)),
            float(dados.get("hornet_y", 0.0)),
            float(dados.get("hornet_vx", 0.0)),
            float(dados.get("hornet_vy", 0.0)),
            vida_mae,
            float(dados.get("mae_x", 0.0)),
            float(dados.get("mae_y", 0.0)),
            float(dados.get("mae_vx", 0.0)),
            float(dados.get("mae_vy", 0.0)),
            float(dados.get("mae_estado_codigo", 0)),
            float(dados.get("arena_min_x", 40.0)),
            float(dados.get("arena_max_x", 75.0)),
            float(dados.get("arena_min_y", 12.0)),
            float(dados.get("arena_max_y", 35.0)),
        ]

        alvos = dados.get("alvos", [])

        # Processa cada alvo calculando Delta X, Delta Y e Distância em relação à Hornet
        lista_alvos_processados = []
        for alvo in alvos:
            pos_x = float(alvo.get("posX", 0.0))
            pos_y = float(alvo.get("posY", 0.0))
            dist = float(alvo.get("distancia", VALOR_PADDING_DISTANCIA))
            delta_x = pos_x - hornet_x
            delta_y = pos_y - hornet_y
            lista_alvos_processados.append((dist, delta_x, delta_y))

        # Ordena alvos por proximidade (os mais perigosos primeiro)
        lista_alvos_processados.sort(key=lambda x: x[0])

        # Extrai os 7 alvos mais próximos
        alvos_selecionados = lista_alvos_processados[:MAX_ALVOS]

        # Monta o array dos alvos (7 * 3 = 21 valores)
        valores_alvos = []
        for dist, dx, dy in alvos_selecionados:
            valores_alvos.extend([dx, dy, dist])

        # Preenchimento (Padding) com [0.0, 0.0, 999.0] para vagas não ocupadas
        alvos_restantes = MAX_ALVOS - len(alvos_selecionados)
        for _ in range(alvos_restantes):
            valores_alvos.extend([0.0, 0.0, VALOR_PADDING_DISTANCIA])

        obs = np.array(dados_base + valores_alvos, dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observação inválida na etapa {self.etapa.ID}: "
                f"esperado {self.observation_space.shape}, recebido {obs.shape}."
            )
        return obs, vida_h, vida_mae

    def _ler_tempo_jogo(self, dados):
        try:
            tempo_jogo = float(dados["tempo_jogo"])
        except (KeyError, TypeError, ValueError) as ex:
            raise RuntimeError(
                "O mod C# não enviou um valor válido de 'tempo_jogo' na telemetria."
            ) from ex
        if not np.isfinite(tempo_jogo):
            raise RuntimeError("O mod C# enviou um valor não finito para 'tempo_jogo'.")
        return tempo_jogo

    def _calcular_recompensa_proximidade(self, dados):
        if self.etapa.ID != "etapa_03_dar_dano":
            return 0.0
        if not (
            dados.get("hornet_detectada", False)
            and dados.get("chefe_detectada", False)
            and dados.get("vida_hornet_valida", False)
            and dados.get("vida_mae_valida", False)
        ):
            return 0.0

        chaves_posicao = ("hornet_x", "hornet_y", "mae_x", "mae_y")
        try:
            hornet_x, hornet_y, chefe_x, chefe_y = (
                float(dados[chave]) for chave in chaves_posicao
            )
            largura_arena = float(dados["arena_max_x"]) - float(dados["arena_min_x"])
            altura_arena = float(dados["arena_max_y"]) - float(dados["arena_min_y"])
        except (KeyError, TypeError, ValueError) as ex:
            raise RuntimeError(
                "A telemetria não contém posições válidas da Hornet, da chefe "
                "e dos limites da arena para calcular a recompensa de proximidade."
            ) from ex

        valores = (hornet_x, hornet_y, chefe_x, chefe_y, largura_arena, altura_arena)
        if not all(math.isfinite(valor) for valor in valores):
            raise RuntimeError(
                "A telemetria contém valores não finitos para a recompensa de proximidade."
            )

        diagonal_arena = math.hypot(largura_arena, altura_arena)
        if diagonal_arena <= 0.0:
            raise RuntimeError(
                "Os limites da arena são inválidos para calcular a proximidade."
            )

        distancia_atual = math.hypot(chefe_x - hornet_x, chefe_y - hornet_y)
        distancia_anterior = self.recompensa_estado.get("distancia_chefe_anterior")
        self.recompensa_estado["distancia_chefe_anterior"] = distancia_atual
        if distancia_anterior is None:
            return 0.0

        progresso = float(distancia_anterior) - distancia_atual
        progresso_limitado = max(-diagonal_arena, min(diagonal_arena, progresso))
        return ESCALA_RECOMPENSA_PROXIMIDADE * progresso_limitado / diagonal_arena

    def _calcular_penalidade_parede(self, dados):
        if self.penalidade_parede is None or self.etapa.ID != "etapa_03_dar_dano":
            return 0.0
        if not dados.get("hornet_detectada", False):
            return 0.0

        try:
            posicao_x = float(dados["hornet_x"])
            limite_esquerdo = float(dados["arena_min_x"])
            limite_direito = float(dados["arena_max_x"])
        except (KeyError, TypeError, ValueError) as ex:
            raise RuntimeError(
                "A telemetria não contém posição X da Hornet e limites laterais "
                "válidos para calcular a penalidade de parede."
            ) from ex

        if not all(
            math.isfinite(valor)
            for valor in (posicao_x, limite_esquerdo, limite_direito)
        ):
            raise RuntimeError(
                "A telemetria contém valores não finitos para os limites laterais."
            )
        if limite_esquerdo >= limite_direito:
            raise RuntimeError("Os limites laterais da arena são inválidos.")

        posicao_anterior = self.recompensa_estado.get("posicao_x_hornet_anterior")
        self.recompensa_estado["posicao_x_hornet_anterior"] = posicao_x
        if posicao_anterior is None:
            return 0.0

        cruzou_esquerda = (
            (posicao_anterior <= limite_esquerdo < posicao_x)
            or (posicao_x <= limite_esquerdo < posicao_anterior)
        )
        cruzou_direita = (
            (posicao_anterior <= limite_direito < posicao_x)
            or (posicao_x <= limite_direito < posicao_anterior)
        )
        cruzamentos = int(cruzou_esquerda) + int(cruzou_direita)
        return self.penalidade_parede * cruzamentos

    def _registrar_telemetria(self, dados, acao, recompensa=0.0):
        try:
            registro = dict(dados)
            registro["etapa"] = self.etapa.ID
            registro["step"] = self.episode_steps
            registro["acao"] = [int(x) for x in acao] if hasattr(acao, "__iter__") else int(acao)
            registro["recompensa_step"] = round(float(recompensa), 4)

            with open(self.arquivo_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro) + "\n")
        except (OSError, TypeError, ValueError) as ex:
            raise RuntimeError(
                f"Falha ao registrar telemetria em '{self.arquivo_log}': {ex}"
            ) from ex

    def step(self, action):
        self.episode_steps += 1

        # 1. Aplica as teclas reais no teclado (sem viés de injeção física)
        self._aplicar_inputs_teclado(action)

        # 2. Lê o estado resultante em tempo real (transmitido periodicamente pelo mod C#)
        dados = self._ler_dados_tcp()

        obs, _, _ = self._construir_observacao(dados)
        tempo_decorrido_jogo = None
        derrota_por_tempo = False
        if self.max_episode_seconds is not None:
            if self._tempo_jogo_inicio is None:
                raise RuntimeError(
                    "O episódio não possui tempo inicial do jogo; execute reset() antes de step()."
                )
            tempo_atual_jogo = self._ler_tempo_jogo(dados)
            tempo_decorrido_jogo = tempo_atual_jogo - self._tempo_jogo_inicio
            if tempo_decorrido_jogo < 0:
                raise RuntimeError(
                    "O relógio do jogo retrocedeu durante a avaliação do episódio."
                )
            derrota_por_tempo = tempo_decorrido_jogo >= self.max_episode_seconds

        vida_hornet_bruta = float(dados.get("vida_hornet", -1))
        vida_mae_bruta = float(dados.get("vida_mae", -1))
        vitoria_confirmada = bool(dados.get("vitoria_real_confirmada", False))
        if vitoria_confirmada and vida_mae_bruta < 0:
            vida_mae_bruta = 0.0
        vida_hornet_valida = (
            bool(dados.get("vida_hornet_valida", vida_hornet_bruta >= 0))
            and vida_hornet_bruta >= 0
        )
        vida_mae_valida = (
            bool(dados.get("vida_mae_valida", vida_mae_bruta >= 0))
            or vitoria_confirmada
        )
        morreu_realmente = (
            vida_hornet_valida
            and bool(dados.get("hornet_detectada", True))
            and bool(
                dados.get(
                    "morte_real_confirmada",
                    vida_hornet_bruta == 0 or bool(dados.get("morto", False)),
                )
            )
        )
        morreu = morreu_realmente or derrota_por_tempo
        chefe_sem_vida = vida_mae_valida and vida_mae_bruta <= 0
        limite_hp_chefe = (
            self.etapa.ID == "etapa_03_dar_dano"
            and vida_mae_valida
            and vida_mae_bruta > 0
            and vida_mae_bruta <= self.etapa.LIMIAR_HP_CHEFE_EPISODIO
            and not morreu
        )
        venceu = (
            (chefe_sem_vida or (vitoria_confirmada and not vida_mae_valida))
            and not morreu
            and not limite_hp_chefe
        )
        limite_steps = (
            self.episode_steps >= self.max_episode_steps
            and not morreu
            and not venceu
            and not limite_hp_chefe
        )
        done = morreu or venceu or limite_hp_chefe
        truncated = limite_steps
        if derrota_por_tempo and not morreu_realmente:
            self._motivo_ultimo_termino = "limite_tempo"
            print(
                f"[DERROTA POR TEMPO] Limite de "
                f"{self.max_episode_seconds:.0f}s de tempo do jogo atingido."
            )
        elif morreu_realmente:
            self._motivo_ultimo_termino = "morte"
            print(
                f"[MORTE CONFIRMADA NO STEP {self.episode_steps}] "
                "Encerrando episódio..."
            )
        elif limite_hp_chefe:
            self._motivo_ultimo_termino = "limite_hp_chefe"
            print(
                f"[LIMITE DE HP DA CHEFE: {vida_mae_bruta:.0f}/"
                f"{self.etapa.HP_CHEFE_MAX} NO STEP {self.episode_steps}] "
                "Encerrando episódio para restaurar a chefe."
            )
        elif venceu:
            self._motivo_ultimo_termino = "vitoria"
            print(
                f"[VITÓRIA CONFIRMADA NO STEP {self.episode_steps}] "
                "Mãe Musgo derrotada!"
            )
        elif limite_steps:
            self._motivo_ultimo_termino = "limite_steps"
            print(
                f"[LIMITE DE STEPS DA ETAPA {self.etapa.ID} NO STEP "
                f"{self.episode_steps}] Encerrando episódio..."
            )

        hp_chefe_anterior = self.recompensa_estado["hp_chefe_anterior"]
        hp_hornet_anterior = self.recompensa_estado["hp_hornet_anterior"]
        dano_causado = (
            max(0, hp_chefe_anterior - vida_mae_bruta)
            if vida_mae_valida and hp_chefe_anterior is not None
            else 0
        )
        dano_recebido = (
            max(0, hp_hornet_anterior - vida_hornet_bruta)
            if vida_hornet_valida and hp_hornet_anterior is not None
            else 0
        )
        cura_realizada = (
            max(0, vida_hornet_bruta - hp_hornet_anterior)
            if vida_hornet_valida and hp_hornet_anterior is not None
            else 0
        )
        if vida_mae_valida:
            self.recompensa_estado["hp_chefe_anterior"] = vida_mae_bruta
        if vida_hornet_valida:
            self.recompensa_estado["hp_hornet_anterior"] = vida_hornet_bruta
        self.recompensa_estado["dano_total"] += dano_causado
        self.recompensa_estado["golpes_recebidos"] += dano_recebido

        argumentos_recompensa = {
            "dano_causado": dano_causado,
            "dano_recebido": dano_recebido,
            "cura_realizada": cura_realizada,
            "morreu": morreu,
            "venceu": venceu,
            "vida_hornet": vida_hornet_bruta if vida_hornet_valida else None,
        }
        recompensa_proximidade = self._calcular_recompensa_proximidade(dados)
        penalidade_parede = self._calcular_penalidade_parede(dados)
        if self.etapa.ID == "etapa_03_dar_dano":
            argumentos_recompensa["recompensa_proximidade"] = recompensa_proximidade
            if self.penalidade_tempo is not None:
                argumentos_recompensa["penalidade_tempo"] = self.penalidade_tempo
            argumentos_recompensa["penalidade_parede"] = penalidade_parede

        recompensa, info_recompensa = self.etapa.calcular_recompensa(
            **argumentos_recompensa,
        )
        self.recompensa_estado["recompensa_acumulada_episodio"] += recompensa
        info_recompensa["etapa"] = self.etapa.ID
        info_recompensa["dano_causado"] = dano_causado
        info_recompensa["dano_recebido"] = dano_recebido
        info_recompensa["cura_realizada"] = cura_realizada
        info_recompensa["recompensa_proximidade"] = recompensa_proximidade
        info_recompensa["penalidade_parede"] = -penalidade_parede
        info_recompensa["derrota_por_tempo"] = derrota_por_tempo
        info_recompensa["limite_hp_chefe"] = limite_hp_chefe
        if tempo_decorrido_jogo is not None:
            info_recompensa["tempo_decorrido_jogo"] = tempo_decorrido_jogo

        # Ao finalizar o episódio, libera as teclas e injeta as métricas no info para o callback gerar o gráfico
        if done or truncated:
            self._liberar_todas_teclas()
            info_recompensa["episode"] = {
                "r": float(self.recompensa_estado["recompensa_acumulada_episodio"]),
                "l": int(self.episode_steps),
            }

        # Registra telemetria
        self._registrar_telemetria(dados, action, recompensa)

        return obs, recompensa, done, truncated, info_recompensa

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._liberar_todas_teclas()
        self.episode_steps = 0
        self.recompensa_estado = inicializar_recompensa()

        self._reset_id += 1
        if not self._sessao_iniciada:
            tipo_inicio = "START" if self.preservar_estado_inicial else "INIT"
            comando = f"{tipo_inicio}:{self._reset_id}:{self.etapa.ID}"
            print(
                "\n[Ambiente] >>> "
                + (
                    "INICIANDO COM ESTADO ATUAL"
                    if self.preservar_estado_inicial
                    else "INICIALIZANDO MOD"
                )
                + " E AGUARDANDO ESTADO VÁLIDO <<<"
            )
        else:
            if self._motivo_ultimo_termino not in {
                "morte",
                "vitoria",
                "limite_tempo",
                "limite_steps",
                "limite_hp_chefe",
            }:
                raise RuntimeError(
                    "Reset recusado: o episódio não terminou por morte, vitória "
                    "ou limite de tempo."
                )
            comando = (
                f"RESET:{self._reset_id}:"
                f"{self._motivo_ultimo_termino.upper()}:{self.etapa.ID}"
            )
            print("\n[Ambiente] >>> AGUARDANDO CONFIRMAÇÃO DO RESPAWN <<<")

        timeout_anterior = self.conn.gettimeout()
        self.conn.settimeout(20.0)
        try:
            self._enviar_comando_tcp(comando)
            while True:
                dados = self._ler_proxima_mensagem_tcp()
                if dados.get("reset_id") != self._reset_id:
                    continue
                if dados.get("reset_rejeitado"):
                    raise RuntimeError(
                        f"Mod C# rejeitou o reset {self._reset_id}: "
                        f"{dados.get('motivo_reset', 'motivo não informado')}"
                    )
                if dados.get("reset_completo"):
                    break
        finally:
            self.conn.settimeout(timeout_anterior)

        if not dados.get("vida_hornet_valida", False) or dados.get("vida_hornet", -1) <= 0:
            raise RuntimeError("Reset confirmado pelo mod, mas a vida da Hornet não é válida.")

        self._sessao_iniciada = True
        self._motivo_ultimo_termino = None
        self._tempo_jogo_inicio = (
            self._ler_tempo_jogo(dados)
            if self.max_episode_seconds is not None
            else None
        )
        if dados.get("vida_mae_valida", False) and dados.get("vida_mae", -1) >= 0:
            self.recompensa_estado["hp_chefe_anterior"] = float(dados["vida_mae"])
        self.recompensa_estado["hp_hornet_anterior"] = int(dados["vida_hornet"])

        obs, _, _ = self._construir_observacao(dados)

        self._registrar_telemetria(dados, [0] * 7, 0.0)

        print("[Ambiente] >>> RESET CONCLUÍDO COM SUCESSO! <<<")
        return obs, {}

    def close(self):
        self._liberar_todas_teclas()
        try:
            self.conn.close()
            self.server.close()
        except OSError:
            pass
