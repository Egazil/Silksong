import json
import socket
import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.0
except ImportError:
    pydirectinput = None

# Mapeamento dos botões (Ações MultiBinary de tamanho 6)
# 0: Esquerda, 1: Direita, 2: Cima, 3: Baixo, 4: Pular, 5: Atacar
MAPA_TECLAS = {
    0: "left",   # Esquerda
    1: "right",  # Direita
    2: "up",     # Cima (Olhar/Ataque para cima)
    3: "down",   # Baixo (Olhar/Pogo para baixo)
    4: "z",      # Pulo
    5: "x",      # Ataque
}

NOMES_ACOES = [
    "ESQUERDA",
    "DIREITA",
    "CIMA",
    "BAIXO",
    "PULAR",
    "ATACAR",
]

MAX_ALVOS = 7
VALOR_PADDING_DISTANCIA = 999.0


def inicializar_recompensa():
    return {
        "dano_total": 0,
        "golpes_recebidos": 0,
        "hp_chefe_anterior": None,
        "hp_hornet_anterior": None,
        "recompensa_acumulada_episodio": 0.0,
    }


def calcular_recompensa(
    estado,
    hp_chefe,
    hp_hornet,
    hp_chefe_max=120,
    terminado=False,
    vitoria=False,
):
    """
    Função de Recompensa focada em DPS e Sobrevivência.
    """
    if estado["hp_chefe_anterior"] is None:
        estado["hp_chefe_anterior"] = hp_chefe
        estado["hp_hornet_anterior"] = hp_hornet

    # 1. Dano causado ao chefe escalonado pela vida da Hornet
    dano_causado = max(0, estado["hp_chefe_anterior"] - hp_chefe)
    estado["dano_total"] += dano_causado
    multiplicador_vida = max(0.1, hp_hornet / 5.0)
    recompensa_dano = ((dano_causado / hp_chefe_max) * 20.0) * multiplicador_vida

    # 2. Penalidade Constante de Tempo
    penalidade_tempo = 0.01

    # 3. Penalidade por Golpe Sofrido
    dano_recebido = max(0, estado["hp_hornet_anterior"] - hp_hornet)
    if dano_recebido > 0:
        estado["golpes_recebidos"] += dano_recebido
    penalidade_golpe = float(dano_recebido * 1.0)

    # 4. Bônus de Vitória
    bonus_vitoria = 0.0
    if terminado and vitoria:
        bonus_vitoria = 30.0 + (10.0 * (max(0, hp_hornet) / 5.0))

    recompensa_step = (
        recompensa_dano
        - penalidade_tempo
        - penalidade_golpe
        + bonus_vitoria
    )

    estado["hp_chefe_anterior"] = hp_chefe
    estado["hp_hornet_anterior"] = hp_hornet
    estado["recompensa_acumulada_episodio"] += recompensa_step

    if terminado:
        print(
            f"📊 [FIM DE LUTA] Recompensa Total: {estado['recompensa_acumulada_episodio']:.2f} | "
            f"Dano Causado: {estado['dano_total']}/{hp_chefe_max} | Golpes Tomados: {estado['golpes_recebidos']}"
        )

    return recompensa_step, {
        "dano": recompensa_dano,
        "tempo": penalidade_tempo,
        "golpe": penalidade_golpe,
        "vitoria": bonus_vitoria,
    }


class SilksongEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="127.0.0.1", port=5001):
        super(SilksongEnv, self).__init__()

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
        self.teclas_pressionadas = {i: False for i in range(6)}
        self.arquivo_log = "auditoria_telemetria.jsonl"
        print(f"[Telemetria] Registrando dados e inputs em: '{self.arquivo_log}'")

        # 6 Ações Independentes (MultiBinary):
        # [Esquerda, Direita, Cima, Baixo, Pular, Atacar]
        self.action_space = spaces.MultiBinary(6)

        # Observation Space (26 neurônios):
        # 5 dados base: [vida_hornet, pode_focar, hornet_x, hornet_y, vida_mae]
        # + 21 dados de alvos: 7 alvos * [delta_x, delta_y, distancia]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
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
        Lê do buffer TCP de forma contínua e sem timeouts.
        Drena todos os frames acumulados para entregar sempre o estado mais recente em tempo real.
        """
        while "\n" not in self.buffer:
            chunk = self.conn.recv(4096).decode("utf-8", errors="ignore")
            if not chunk:
                raise ConnectionError("Conexão perdida com o mod C# do Silksong.")
            self.buffer += chunk

        ultimo_dados = None
        while "\n" in self.buffer:
            linha, self.buffer = self.buffer.split("\n", 1)
            linha_str = linha.strip()
            if linha_str:
                try:
                    ultimo_dados = json.loads(linha_str)
                except json.JSONDecodeError:
                    continue

        if ultimo_dados is not None:
            return ultimo_dados

        return self._ler_dados_tcp()

    def _construir_observacao(self, dados):
        """
        Gera o vetor de 26 valores com padding para alvos ausentes.
        """
        vida_h = float(dados.get("vida_hornet", 0))
        pode_focar = float(1.0 if dados.get("pode_focar", False) else 0.0)
        hornet_x = float(dados.get("hornet_x", 0.0))
        hornet_y = float(dados.get("hornet_y", 0.0))
        vida_mae = float(dados.get("vida_mae", 120))
        if vida_mae < 0:
            vida_mae = 120.0

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

        # 5 dados base + 21 dados de alvos = 26 elementos
        obs = np.array(
            [vida_h, pode_focar, hornet_x, hornet_y, vida_mae] + valores_alvos,
            dtype=np.float32,
        )
        return obs, vida_h, vida_mae

    def _registrar_telemetria(self, dados, acao, recompensa=0.0):
        try:
            registro = dict(dados)
            registro["step"] = self.episode_steps
            registro["acao"] = [int(x) for x in acao] if hasattr(acao, "__iter__") else int(acao)
            registro["recompensa_step"] = round(float(recompensa), 4)

            with open(self.arquivo_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(registro) + "\n")
        except Exception:
            pass

    def step(self, action):
        self.episode_steps += 1

        # 1. Aplica as teclas reais no teclado (sem viés de injeção física)
        self._aplicar_inputs_teclado(action)

        # 2. Lê o estado resultante em tempo real (transmitido periodicamente pelo mod C#)
        dados = self._ler_dados_tcp()

        obs, vida_h, vida_mae = self._construir_observacao(dados)
        morto_mod = bool(dados.get("morto", False))

        done = False
        vitoria = False

        hp_chefe_anterior = self.recompensa_estado.get("hp_chefe_anterior")
        HP_MAX_CHEFE = 120
        chefe_resetou = (
            hp_chefe_anterior is not None
            and hp_chefe_anterior < HP_MAX_CHEFE
            and vida_mae >= HP_MAX_CHEFE
        )

        if vida_h <= 0 or morto_mod or chefe_resetou:
            print(f"❌ [MORTE NO STEP {self.episode_steps}] Encerrando episódio...")
            done = True
            vitoria = False
        elif vida_mae <= 0:
            print(f"🏆 [VITÓRIA NO STEP {self.episode_steps}] Mãe Musgo derrotada!")
            done = True
            vitoria = True

        recompensa, info_recompensa = calcular_recompensa(
            estado=self.recompensa_estado,
            hp_chefe=vida_mae,
            hp_hornet=vida_h,
            hp_chefe_max=HP_MAX_CHEFE,
            terminado=done,
            vitoria=vitoria,
        )

        # Ao finalizar o episódio, libera as teclas e injeta as métricas no info para o callback gerar o gráfico
        if done:
            self._liberar_todas_teclas()
            info_recompensa["episode"] = {
                "r": float(self.recompensa_estado["recompensa_acumulada_episodio"]),
                "l": int(self.episode_steps),
            }

        # Registra telemetria
        self._registrar_telemetria(dados, action, recompensa)

        return obs, recompensa, done, False, info_recompensa

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._liberar_todas_teclas()
        self.episode_steps = 0
        self.recompensa_estado = inicializar_recompensa()

        print("\n[Ambiente] >>> ENVIANDO COMANDO DE RESET PARA O MOD C# <<<")
        self.buffer = ""
        self._enviar_comando_tcp("99")

        dados = self._ler_dados_tcp()

        self.recompensa_estado["hp_chefe_anterior"] = float(dados.get("vida_mae", 120))
        self.recompensa_estado["hp_hornet_anterior"] = int(dados.get("vida_hornet", 5))

        obs, _, _ = self._construir_observacao(dados)

        self._registrar_telemetria(dados, [0] * 6, 0.0)

        print("[Ambiente] >>> RESET CONCLUÍDO COM SUCESSO! <<<")
        return obs, {}

    def close(self):
        self._liberar_todas_teclas()
        try:
            self.conn.close()
            self.server.close()
        except Exception:
            pass
