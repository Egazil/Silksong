import time
from pynput.keyboard import Controller, Key

keyboard = Controller()

print("--- INICIANDO TESTE DE CONTROLE DO SILKSONG ---")
print(
    "ATENÇÃO: Abra o jogo e dê foco na janela em 3 segundos para começar a"
    " teste!"
)
time.sleep(3)


def testar_sequencia_basica():
  print("1. Movendo para a ESQUERDA...")
  keyboard.press(Key.left)  # Segura a seta para a esquerda
  time.sleep(1.0)  # Anda por 1 segundo

  print("2. Pulando (Z) e mantendo a esquerda...")
  keyboard.press("z")  # Pula
  time.sleep(0.1)
  keyboard.release("z")

  # Pequeno atraso no ar para executar o ataque logo em seguida
  time.sleep(0.2)

  print("3. Atacando (X)...")
  keyboard.press("x")  # Ataca
  time.sleep(0.1)
  keyboard.release("x")

  # Espera o pulo terminar enquanto anda
  time.sleep(0.5)

  print("4. Parando de andar para a esquerda.")
  keyboard.release(Key.left)


if __name__ == "__main__":
  try:
    testar_sequencia_basica()
    print("Teste finalizado com sucesso!")
  except KeyboardInterrupt:
    print("Teste interrompido pelo usuário.")
    # Garante que solta as teclas se você parar o script no meio
    keyboard.release(Key.left)
    keyboard.release("z")
    keyboard.release("x")