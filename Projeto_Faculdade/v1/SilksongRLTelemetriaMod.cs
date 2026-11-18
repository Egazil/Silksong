using BepInEx;
using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Globalization;

[BepInPlugin("com.seuusuario.silksongrltelemetria", "Mod RL Telemetria Silksong Periódico", "3.2.0")]
public class SilksongRLTelemetriaMod : BaseUnityPlugin
{
    private Transform hornetTransform;
    private Collider2D hornetCollider;
    private HeroController heroController;
    private Camera cameraPrincipal;

    private int vidaHornet = 0;
    private bool podeFocar = false;
    private Vector2 posicaoHornet = Vector2.zero;

    private int vidaMother = -1;
    private HealthManager motherHealthManager;

    // --- SAVESTATE: Posições de Respawn Instantâneo ---
    private Vector2 spawnHornet = Vector2.zero;
    private Vector2 spawnBoss = Vector2.zero;
    private bool spawnRegistrado = false;

    // Conexão TCP com o Python
    private TcpClient clienteTcp;
    private NetworkStream fluxoRede;
    private float tempoTentativaConexao = 0f;

    // Controle de Transmissão Periódica
    private int contadorFrames = 0;
    private const int INTERVALO_FRAMES = 2; // Transmite a cada 2 frames (~30 vezes por segundo)
    private bool estaEmReset = false;
    private const float RAIO_BUSCA_ALVOS = 25.0f;

    // Bounding Box Visual para Depuração do Raio de Visão
    private Texture2D texturaBorda;
    private GUIStyle estiloTextoDebug;

    private void Awake()
    {
        Logger.LogInfo("Mod RL Telemetria (Transmissão Periódica + Savestate Instantâneo) iniciado!");
        
        // Permite o treinamento mesmo em segundo plano
        Application.runInBackground = true;

        // Velocidade normal do jogo (1x)
        Time.timeScale = 1.0f;

        // Inicializa texturas para desenhar a Bounding Box de visão na tela
        texturaBorda = new Texture2D(1, 1);
        texturaBorda.SetPixel(0, 0, Color.white);
        texturaBorda.Apply();

        estiloTextoDebug = new GUIStyle();
        estiloTextoDebug.normal.textColor = Color.yellow;
        estiloTextoDebug.fontSize = 12;
        estiloTextoDebug.fontStyle = FontStyle.Bold;

        TentarConexaoComPython();
    }

    private void TentarConexaoComPython()
    {
        try
        {
            clienteTcp = new TcpClient();
            clienteTcp.Connect("127.0.0.1", 5001);
            fluxoRede = clienteTcp.GetStream();
            Logger.LogInfo("Conectado com sucesso ao ambiente Python!");
        }
        catch (Exception e)
        {
            Logger.LogWarning($"Aguardando Python abrir a porta 5001... ({e.Message})");
            clienteTcp = null;
            fluxoRede = null;
        }
    }

    private void Update()
    {
        if (cameraPrincipal == null)
        {
            cameraPrincipal = Camera.main;
        }

        if (clienteTcp == null || !clienteTcp.Connected || fluxoRede == null)
        {
            tempoTentativaConexao += Time.deltaTime;
            if (tempoTentativaConexao >= 3.0f)
            {
                tempoTentativaConexao = 0f;
                TentarConexaoComPython();
            }
            return;
        }

        if (hornetTransform == null || heroController == null)
        {
            heroController = UnityEngine.Object.FindFirstObjectByType<HeroController>();
            if (heroController != null)
            {
                hornetTransform = heroController.transform;
                hornetCollider = hornetTransform.GetComponentInChildren<Collider2D>();
            }
        }

        if (hornetTransform != null)
        {
            posicaoHornet = hornetTransform.position;
        }

        // Localiza a Mãe Musgo se ainda não encontrada
        if (motherHealthManager == null)
        {
            Collider2D[] alvos = Physics2D.OverlapCircleAll(posicaoHornet, RAIO_BUSCA_ALVOS);
            foreach (var col in alvos)
            {
                if (col != null && col.gameObject.name.ToLower().Contains("mother"))
                {
                    motherHealthManager = col.GetComponentInChildren<HealthManager>() ?? col.GetComponentInParent<HealthManager>();
                    break;
                }
            }
        }

        ColetarEstadoAtual();

        // 1. TRANSMISSÃO PERIÓDICA: Envia os dados continuamente para o Python
        if (!estaEmReset)
        {
            contadorFrames++;
            if (contadorFrames >= INTERVALO_FRAMES)
            {
                contadorFrames = 0;
                TransmitirJsonParaPython();
            }
        }

        // 2. RECEPÇÃO DE COMANDOS DO PYTHON: Leitura direta de bytes (Sem StreamReader para evitar travamento)
        if (fluxoRede != null && fluxoRede.DataAvailable && !estaEmReset)
        {
            try
            {
                byte[] bufferBytes = new byte[256];
                int bytesLidos = fluxoRede.Read(bufferBytes, 0, bufferBytes.Length);
                if (bytesLidos > 0)
                {
                    string comando = Encoding.UTF8.GetString(bufferBytes, 0, bytesLidos).Trim();
                    if (comando.Contains("99") || comando.ToLower().Contains("reset"))
                    {
                        StartCoroutine(RotinaResetInstantaneo());
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.LogError($"Erro ao ler comando do Python: {ex.Message}");
                Desconectar();
            }
        }
    }

    private void ColetarEstadoAtual()
    {
        try
        {
            vidaHornet = PlayerData.instance.GetInt("health");
            int sedaAtual = PlayerData.instance.GetInt("silk");
            podeFocar = (sedaAtual > 0);
        }
        catch
        {
            vidaHornet = 0;
            podeFocar = false;
        }

        if (motherHealthManager != null)
        {
            vidaMother = motherHealthManager.hp;
        }
        else
        {
            vidaMother = -1;
        }
    }

    private string LimparTextoJson(string texto)
    {
        if (string.IsNullOrEmpty(texto)) return "";
        return texto.Replace("\"", "").Replace("\\", "").Replace("\n", "").Replace("\r", "").Trim();
    }

    private void TransmitirJsonParaPython()
    {
        if (fluxoRede == null || clienteTcp == null || !clienteTcp.Connected) return;

        try
        {
            List<string> listaAlvosJson = new List<string>();
            Collider2D[] colisoresProximos = Physics2D.OverlapCircleAll(posicaoHornet, RAIO_BUSCA_ALVOS);

            foreach (var col in colisoresProximos)
            {
                if (col == null || !col.gameObject.activeInHierarchy) continue;

                string nomeObj = col.gameObject.name;
                string nomeLower = nomeObj.ToLower();

                if (nomeLower.Contains("mother") || nomeLower.Contains("stalactite") || nomeLower.Contains("crawler"))
                {
                    if (nomeLower.Contains("mother") && motherHealthManager == null)
                    {
                        motherHealthManager = col.GetComponentInChildren<HealthManager>() ?? col.GetComponentInParent<HealthManager>();
                    }

                    float dist = 0f;
                    if (hornetCollider != null)
                    {
                        dist = (float)Math.Round(Vector2.Distance(hornetCollider.bounds.center, col.bounds.center), 2);
                    }

                    float posX = (float)Math.Round(col.bounds.center.x, 2);
                    float posY = (float)Math.Round(col.bounds.center.y, 2);

                    string posXStr = posX.ToString(CultureInfo.InvariantCulture);
                    string posYStr = posY.ToString(CultureInfo.InvariantCulture);
                    string distStr = dist.ToString(CultureInfo.InvariantCulture);

                    string nomeLimpo = LimparTextoJson(nomeObj);
                    string alvoJson = $"{{\"nome\":\"{nomeLimpo}\",\"posX\":{posXStr},\"posY\":{posYStr},\"distancia\":{distStr}}}";
                    listaAlvosJson.Add(alvoJson);
                }
            }

            string alvosArrayStr = "[" + string.Join(",", listaAlvosJson) + "]";

            float hX = (float)Math.Round(posicaoHornet.x, 2);
            float hY = (float)Math.Round(posicaoHornet.y, 2);

            string hXStr = hX.ToString(CultureInfo.InvariantCulture);
            string hYStr = hY.ToString(CultureInfo.InvariantCulture);
            string focarStr = podeFocar ? "true" : "false";
            string mortoStr = vidaHornet <= 0 ? "true" : "false";

            string jsonString = $"{{\"vida_hornet\":{vidaHornet},\"pode_focar\":{focarStr},\"hornet_x\":{hXStr},\"hornet_y\":{hYStr},\"vida_mae\":{vidaMother},\"morto\":{mortoStr},\"alvos\":{alvosArrayStr}}}\n";

            byte[] dadosBytes = Encoding.UTF8.GetBytes(jsonString);
            fluxoRede.Write(dadosBytes, 0, dadosBytes.Length);
            fluxoRede.Flush();
        }
        catch (Exception e)
        {
            Logger.LogError($"Erro ao transmitir JSON: {e.Message}");
            Desconectar();
        }
    }

    // --- VISUALIZAÇÃO DO RADAR (BOUNDING BOX NA TELA) ---
    private void OnGUI()
    {
        if (cameraPrincipal == null || hornetTransform == null) return;

        Vector3 cantoSuperiorEsquerdoMundo = new Vector3(posicaoHornet.x - RAIO_BUSCA_ALVOS, posicaoHornet.y + RAIO_BUSCA_ALVOS, 0f);
        Vector3 cantoInferiorDireitoMundo = new Vector3(posicaoHornet.x + RAIO_BUSCA_ALVOS, posicaoHornet.y - RAIO_BUSCA_ALVOS, 0f);

        Vector3 screenCantoSupEsq = cameraPrincipal.WorldToScreenPoint(cantoSuperiorEsquerdoMundo);
        Vector3 screenCantoInfDir = cameraPrincipal.WorldToScreenPoint(cantoInferiorDireitoMundo);

        float x = screenCantoSupEsq.x;
        float y = Screen.height - screenCantoSupEsq.y;
        float largura = screenCantoInfDir.x - screenCantoSupEsq.x;
        float altura = screenCantoSupEsq.y - screenCantoInfDir.y;

        if (screenCantoSupEsq.z > 0 && largura > 0 && altura > 0)
        {
            Rect rectBoundingBox = new Rect(x, y, largura, altura);
            DesenharContornoRetangulo(rectBoundingBox, Color.yellow, 2f);
            GUI.Label(new Rect(x + 5, y + 5, 250, 20), $"Bounding Box Radar IA (Raio: {RAIO_BUSCA_ALVOS})", estiloTextoDebug);
        }
    }

    private void DesenharContornoRetangulo(Rect rect, Color cor, float espessuraLinha)
    {
        Color corAntiga = GUI.color;
        GUI.color = cor;
        GUI.DrawTexture(new Rect(rect.x, rect.y, rect.width, espessuraLinha), texturaBorda);
        GUI.DrawTexture(new Rect(rect.x, rect.y + rect.height - espessuraLinha, rect.width, espessuraLinha), texturaBorda);
        GUI.DrawTexture(new Rect(rect.x, rect.y, espessuraLinha, rect.height), texturaBorda);
        GUI.DrawTexture(new Rect(rect.x + rect.width - espessuraLinha, rect.y, espessuraLinha, rect.height), texturaBorda);
        GUI.color = corAntiga;
    }

    // --- ROTINA DE RESPAWN INSTANTÂNEO (SAVESTATE SEM TIMEOUTS DE 5 SEGUNDOS) ---
    private IEnumerator RotinaResetInstantaneo()
    {
        estaEmReset = true;

        // PRIMEIRA EXECUÇÃO (Disparada quando o usuário pressiona ENTER no Python):
        if (!spawnRegistrado)
        {
            if (hornetTransform != null) spawnHornet = hornetTransform.position;
            if (motherHealthManager != null) spawnBoss = motherHealthManager.transform.position;
            spawnRegistrado = true;

            try
            {
                PlayerData.instance.SetInt("health", 5);
                if (motherHealthManager != null) motherHealthManager.hp = 120;
            }
            catch { }

            Logger.LogInfo($"[Savestate] Inicialização realizada! Posições salvas -> Hornet: {spawnHornet} | Chefe: {spawnBoss}.");

            ColetarEstadoAtual();
            TransmitirJsonParaPython();

            estaEmReset = false;
            yield break;
        }

        // EXECUÇÕES SUBSEQUENTES (Pausa de 5 segundos para estabilização das animações de morte):
        Logger.LogInfo("[Savestate] Morte detectada! Aguardando 5 segundos para estabilização das animações...");
        yield return new WaitForSeconds(5.0f);

        Logger.LogInfo("[Savestate] Executando Respawn Instantâneo...");

        // 1. Limpeza de Arena: Destrói invocações (crawlers) e espinhos/projéteis caídos
        try
        {
            Collider2D[] todosColisores = UnityEngine.Object.FindObjectsByType<Collider2D>(FindObjectsSortMode.None);
            foreach (var col in todosColisores)
            {
                if (col == null || col.gameObject == null) continue;
                string nome = col.gameObject.name.ToLower();
                if (nome.Contains("crawler") || nome.Contains("stalactite"))
                {
                    UnityEngine.Object.Destroy(col.gameObject);
                }
            }
        }
        catch (Exception ex)
        {
            Logger.LogWarning($"[Savestate] Falha ao limpar arena: {ex.Message}");
        }

        // 2. Teleporta a Hornet e reseta física e vida
        if (hornetTransform != null)
        {
            hornetTransform.position = spawnHornet;
            var rb = hornetTransform.GetComponent<Rigidbody2D>();
            if (rb != null)
            {
                rb.linearVelocity = Vector2.zero;
            }
        }

        try
        {
            PlayerData.instance.SetInt("health", 5);
            PlayerData.instance.SetInt("silk", 0);
        }
        catch { }

        // 3. Teleporta o Chefe, enche o HP e reinicia o FSM (SetActive false -> true)
        if (motherHealthManager != null && motherHealthManager.gameObject != null)
        {
            var bossObj = motherHealthManager.gameObject;
            bossObj.transform.position = spawnBoss;
            motherHealthManager.hp = 120;

            bossObj.SetActive(false);
            yield return null; // Aguarda 1 frame
            bossObj.SetActive(true);
        }

        yield return null; // Apenas 1 frame de estabilização

        ColetarEstadoAtual();
        TransmitirJsonParaPython();

        estaEmReset = false;
        Logger.LogInfo("[Savestate] Respawn Instantâneo concluído com sucesso!");
    }

    private void Desconectar()
    {
        try
        {
            if (fluxoRede != null) fluxoRede.Close();
            if (clienteTcp != null) clienteTcp.Close();
        }
        catch { }
        clienteTcp = null;
        fluxoRede = null;
    }

    private void OnApplicationQuit()
    {
        Desconectar();
    }
}
