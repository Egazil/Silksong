import socket
import time

HOST = "127.0.0.1"
PORT = 5001

print("=" * 65)
print("🕹️  TESTADOR MANUAL DE RESPAWN INSTANTÂNEO (SAVESTATE)")
print("=" * 65)
print(f"Iniciando servidor de teste na porta {PORT}...")
print("Abra o jogo com o Mod instalado e entre na arena da Mãe Musgo.")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

conn, addr = server.accept()
print(f"\n✅ Mod C# conectado com sucesso através de {addr}!")
print("O jogo agora está sincronizado com este script de teste.")
print("\n" + "-" * 65)
print("INSTRUÇÕES:")
print("1. Deixe a Mãe Musgo atacar e invocar filhotes (crawlers).")
print("2. Pressione ENTER neste terminal a qualquer momento para disparar o RESPAWN INSTANTÂNEO.")
print("3. Digite 'sair' para encerrar o teste.")
print("-" * 65 + "\n")

try:
    while True:
        comando = input("👉 Pressione [ENTER] para testar o Respawn (ou 'sair'): ").strip()
        if comando.lower() == "sair":
            break

        inicio = time.time()
        # Envia o comando 99 (Reset) para o mod C#
        conn.sendall(b"99\n")

        # Aguarda a resposta do C# confirmando que a arena foi limpa e resetada
        resposta = conn.recv(4096).decode("utf-8")
        delta = (time.time() - inicio) * 1000

        print(f"⚡ [RESPAWN EXECUTADO EM {delta:.1f} ms!]")
        print(f"📦 Estado pós-reset retornado pelo C#:\n{resposta.strip()}\n")

except KeyboardInterrupt:
    print("\nTeste interrompido pelo usuário.")
finally:
    conn.close()
    server.close()
    print("Conexão encerrada.")
