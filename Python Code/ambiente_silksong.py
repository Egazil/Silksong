import json
import socket
import threading
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pynput.keyboard import Controller, Key


def inicializar_recompensa():
  return {
      "dano_total": 0,
      "golpes_recebidos": 0,
      "tempo_total": 0,
      "hp_chefe_anterior": None,
      "hp_hornet_anterior": None,
      "combate_iniciado": False,
  }


def calcular_recompensa(
    estado,
    hp_chefe,
    hp_hornet,
    hp_chefe_max=120,
    delta_time=0.7,
    terminado=False,
    vitoria=False,
    max_time=120,
    lambda_hit=0.2,
    lambda_time=0.1,
    victory_bonus=10.0,
    time_bonus_weight=2.0,
):
  if estado["hp_chefe_anterior"] is None:
    estado["hp_chefe_anterior"] = hp_chefe
    estado["hp_hornet_anterior"] = hp_hornet

  dano = max(0, estado["hp_chefe_anterior"] - hp_chefe)

  if dano > 0 and not estado["combate_iniciado"]:
    estado["combate_iniciado"] = True

  estado["dano_total"] += dano
  recompensa_dano = 2 * dano / hp_chefe_max

  golpe = hp_hornet < estado["hp_hornet_anterior"]
  if golpe:
    estado["golpes_recebidos"] += 1
  penalidade_golpe = lambda_hit if golpe else 0

  if estado["combate_iniciado"]:
    estado["tempo_total"] += delta_time
    penalidade_tempo = lambda_time * delta_time / max_time
  else:
    penalidade_tempo = 0.0

  recompensa = recompensa_dano - penalidade_golpe - penalidade_tempo

  bonus_vitoria = 0
  bonus_tempo = 0

  if terminado and vitoria:
    bonus_vitoria = victory_bonus
    if estado["combate_iniciado"] and estado["tempo_total"] > 0:
      bonus_tempo = time_bonus_weight * max(
          0, 1 - estado["tempo_total"] / max_time
      )

  recompensa += bonus_vitoria + bonus_tempo

  estado["hp_chefe_anterior"] = hp_chefe
  estado["hp_hornet_anterior"] = hp_hornet

  return recompensa, {
      "dano": recompensa_dano,
      "golpe": penalidade_golpe,
      "tempo": penalidade_tempo,
      "vitoria": bonus_vitoria,
      "velocidade": bonus_tempo,
  }


class SilksongEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(self, host="127.0.0.1", port=5001):
    super(SilksongEnv, self).__init__()

    self.keyboard = Controller()
    self._input_lock = threading.RLock()
    self._movimento_reset_concluido = True

    self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.server.bind((host, port))
    self.server.listen(1)

    print(
        f"Aguardando conexão do mod C# no IP {host}:{port} (Abra o jogo e entre"
        " na arena)..."
    )
    self.conn, self.addr = self.server.accept()
    self.conn.setblocking(True)
    print(f"Mod conectado com sucesso por {self.addr}!")

    self.buffer = ""
    self.recompensa_estado = inicializar_recompensa()
    self.episode_steps = 0

    self.action_space = spaces.Discrete(6)
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
    )

  def _ler_dados_tcp(self):
    while True:
      while "\n" not in self.buffer:
        chunk = self.conn.recv(4096).decode("utf-8", errors="ignore")
        if not chunk:
          raise ConnectionError("Conexão perdida com o mod C# do Silksong.")
        self.buffer += chunk

      linha, self.buffer = self.buffer.split("\n", 1)
      json_str = linha.strip()
      if json_str:
        try:
          dados = json.loads(json_str)
          print(
              f"[JSON REAL] Vida Hornet: {dados.get('vida_hornet')} | Chefe HP:"
              f" {dados.get('vida_mae')}"
          )
          return dados
        except json.JSONDecodeError:
          continue
    return None

  def step(self, action):
    self.episode_steps += 1

    self._executar_acao_teclado(action)
    time.sleep(0.7)

    dados = self._ler_dados_tcp()

    vida_h = int(dados.get("vida_hornet", 0))
    vida_mae = float(dados.get("vida_mae", 120))
    if vida_mae < 0:
      vida_mae = 120

    print(
        f"[STEP {self.episode_steps}] Ação: {action} | Vida Hornet: {vida_h} |"
        f" Chefe HP: {vida_mae}"
    )

    alvos = dados.get("alvos", [])
    menor_dist = 999.0
    for alvo in alvos:
      dist = alvo.get("distancia", 999.0)
      if dist < menor_dist:
        menor_dist = dist

    obs = np.array(
        [
            float(vida_h),
            float(1.0 if dados.get("pode_focar", False) else 0.0),
            float(dados.get("hornet_x", 0.0)),
            float(dados.get("hornet_y", 0.0)),
            float(vida_mae),
            float(len(alvos)),
            float(menor_dist),
        ],
        dtype=np.float32,
    )

    done = False
    vitoria = False

    hp_chefe_anterior = self.recompensa_estado.get("hp_chefe_anterior")
    HP_MAX_CHEFE = 120
    chefe_resetou = (
        hp_chefe_anterior is not None
        and hp_chefe_anterior < HP_MAX_CHEFE
        and vida_mae >= HP_MAX_CHEFE
    )

    if vida_h <= 0 or chefe_resetou:
      print("\n" + "=" * 60)
      print(
          f"❌ [MORTE DETECTADA NO STEP {self.episode_steps}] Limpando inputs e"
          " encerrando episódio..."
      )
      print("=" * 60 + "\n")

      # Limpa imediatamente todos os inputs do teclado ao detectar morte
      self._parar_todos_inputs()

      done = True
      vitoria = False
    elif vida_mae <= 0:
      print("\n" + "=" * 60)
      print(f"🏆 [VITÓRIA NO STEP {self.episode_steps}] Mãe Musgo derrotada!")
      print("=" * 60 + "\n")

      self._parar_todos_inputs()

      done = True
      vitoria = True

    recompensa, info_recompensa = calcular_recompensa(
        estado=self.recompensa_estado,
        hp_chefe=vida_mae,
        hp_hornet=vida_h,
        hp_chefe_max=HP_MAX_CHEFE,
        delta_time=0.7,
        terminado=done,
        vitoria=vitoria,
    )

    return obs, recompensa, done, False, info_recompensa

  def _executar_acao_teclado(self, action):
    with self._input_lock:
      if not self._movimento_reset_concluido:
        raise RuntimeError(
            "A ação do agente foi bloqueada: a caminhada do reset não foi"
            " concluída."
        )

      self._parar_todos_inputs()

      if action == 1:
        self.keyboard.press(Key.left)
      elif action == 2:
        self.keyboard.press(Key.right)
      elif action == 3:
        self.keyboard.press("z")
      elif action == 4:
        self.keyboard.press("x")
      elif action == 5:
        self.keyboard.press("a")

  def _parar_todos_inputs(self):
    """Garante que todas as teclas controladas fiquem soltas."""
    with self._input_lock:
      erro_liberacao = None
      for tecla in (Key.left, Key.right, "z", "x", "a"):
        try:
          self.keyboard.release(tecla)
        except Exception as erro:
          if erro_liberacao is None:
            erro_liberacao = erro

      if erro_liberacao is not None:
        raise RuntimeError("Não foi possível liberar todas as teclas.") from (
            erro_liberacao
        )

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.episode_steps = 0
    self.recompensa_estado = inicializar_recompensa()

    print(
        "\n[Ambiente] >>> RESET: Limpando inputs, aguardando cena recarregar e"
        " andando para a esquerda <<<"
    )

    with self._input_lock:
      self._movimento_reset_concluido = False

      # Limpa todos os inputs antes de iniciar a rotina de caminhada
      self._parar_todos_inputs()

      # Pausa para o Unity processar o recarregamento da cena após a morte
      time.sleep(1.5)

      # Nenhuma ação do agente passa pelo lock até a caminhada terminar.
      self.keyboard.press(Key.left)
      try:
        time.sleep(2.0)
      finally:
        self.keyboard.release(Key.left)

      self._parar_todos_inputs()
      self._movimento_reset_concluido = True

    # Lê o primeiro pacote real pós-reset
    dados = self._ler_dados_tcp()

    self.recompensa_estado["hp_chefe_anterior"] = float(dados.get("vida_mae", 120))
    self.recompensa_estado["hp_hornet_anterior"] = int(dados.get("vida_hornet", 5))

    obs = np.array(
        [
            float(dados.get("vida_hornet", 0)),
            float(1.0 if dados.get("pode_focar", False) else 0.0),
            float(dados.get("hornet_x", 0.0)),
            float(dados.get("hornet_y", 0.0)),
            float(dados.get("vida_mae", 120)),
            float(len(dados.get("alvos", []))),
            999.0,
        ],
        dtype=np.float32,
    )

    return obs, {}

  def close(self):
    try:
      self._parar_todos_inputs()
      self.conn.close()
      self.server.close()
    except Exception:
      pass