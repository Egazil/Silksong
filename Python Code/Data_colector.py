import socket


def escutar_silksong():
  host = "127.0.0.1"
  port = 5001
  print(f"Ouvindo em {host}:{port}...")

  cliente = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  cliente.connect((host, port))

  buffer = ""
  while True:
    dados = cliente.recv(4096).decode("utf-8", errors="ignore")
    if not dados:
      break
    buffer += dados
    while "\n" in buffer:
      linha, buffer = buffer.split("\n", 1)
      linha = linha.strip()
      if linha:
        # AQUI VAMOS IMPRIMIR A LINHA CRUA ENVIADA PELO C#
        print("LINHA CRUA RECEBIDA:")
        print(linha)
        print("-" * 50)


if __name__ == "__main__":
  escutar_silksong()