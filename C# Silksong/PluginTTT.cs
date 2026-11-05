using BepInEx;
using UnityEngine;
using UnityEngine.SceneManagement;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Globalization; // Essencial para forçar o uso de ponto (.) em vez de vírgula (,)

[BepInPlugin("com.seuusuario.silksongrltelemetria", "Mod RL Telemetria Silksong", "1.0.0")]
public class SilksongRLTelemetriaMod : BaseUnityPlugin
{
    private Transform hornetTransform;
    private Collider2D hornetCollider;
    
    private int vidaHornet = 0;
    private bool podeFocar = false;
    private Vector2 posicaoHornet = Vector2.zero;

    private int vidaMother = -1;
    private HealthManager motherHealthManager;

    private TcpClient clienteTcp;
    private NetworkStream fluxoRede;
    private float tempoTentativaConexao = 0f;

    private int contadorFrames = 0;
    private const int INTERVALO_FRAMES = 35;

    private void Awake()
    {
        Logger.LogInfo("Mod de Telemetria iniciado!");
        TentarConexaoComPython();
    }

    private void TentarConexaoComPython()
    {
        try
        {
            clienteTcp = new TcpClient();
            clienteTcp.Connect("127.0.0.1", 5001);
            fluxoRede = clienteTcp.GetStream();
            Logger.LogInfo("Conectado com sucesso ao script Python!");
        }
        catch (Exception e)
        {
            Logger.LogWarning($"Aguardando Python abrir a porta... ({e.Message})");
            clienteTcp = null;
            fluxoRede = null;
        }
    }

    private void Update()
    {
        if (clienteTcp == null || !clienteTcp.Connected)
        {
            tempoTentativaConexao += Time.deltaTime;
            if (tempoTentativaConexao >= 3.0f)
            {
                tempoTentativaConexao = 0f;
                TentarConexaoComPython();
            }
            return;
        }

        try
        {
            vidaHornet = PlayerData.instance.GetInt("health");
            
            // --- SE A HORNET MORREU NO JOGO: REINICIA A CENA ---
            if (vidaHornet <= 0)
            {
                string cenaAtual = SceneManager.GetActiveScene().name;
                SceneManager.LoadScene(cenaAtual);
                return;
            }

            int sedaAtual = PlayerData.instance.GetInt("silk");
            podeFocar = (sedaAtual > 0);
        }
        catch
        {
            vidaHornet = 0;
            podeFocar = false;
        }

        if (hornetTransform == null || hornetCollider == null)
        {
            var hero = UnityEngine.Object.FindFirstObjectByType<HeroController>();
            if (hero != null)
            {
                hornetTransform = hero.transform;
                hornetCollider = hornetTransform.GetComponentInChildren<Collider2D>();
            }
        }

        if (hornetTransform != null) 
        {
            posicaoHornet = hornetTransform.position;
        }

        if (motherHealthManager != null) 
        {
            vidaMother = motherHealthManager.hp;
        }
        else 
        {
            vidaMother = -1;
        }

        contadorFrames++;
        if (contadorFrames >= INTERVALO_FRAMES)
        {
            contadorFrames = 0;
            TransmitirJsonParaPython();
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
            Collider2D[] colisores = UnityEngine.Object.FindObjectsOfType<Collider2D>();

            foreach (var col in colisores)
            {
                if (col == null || !col.gameObject.activeInHierarchy) continue;

                string nomeObj = LimparTextoJson(col.gameObject.name);
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

                    // Formatação segura com ponto decimal universal
                    string posXStr = posX.ToString(CultureInfo.InvariantCulture);
                    string posYStr = posY.ToString(CultureInfo.InvariantCulture);
                    string distStr = dist.ToString(CultureInfo.InvariantCulture);

                    string alvoJson = $"{{\"nome\":\"{nomeObj}\",\"posX\":{posXStr},\"posY\":{posYStr},\"distancia\":{distStr}}}";
                    listaAlvosJson.Add(alvoJson);
                }
            }

            string alvosArrayStr = "[" + string.Join(",", listaAlvosJson) + "]";
            
            float hX = (float)Math.Round(posicaoHornet.x, 2);
            float hY = (float)Math.Round(posicaoHornet.y, 2);
            
            string hXStr = hX.ToString(CultureInfo.InvariantCulture);
            string hYStr = hY.ToString(CultureInfo.InvariantCulture);
            string focarStr = podeFocar ? "true" : "false";

            string jsonString = $"{{\"vida_hornet\":{vidaHornet},\"pode_focar\":{focarStr},\"hornet_x\":{hXStr},\"hornet_y\":{hYStr},\"vida_mae\":{vidaMother},\"alvos\":{alvosArrayStr}}}\n";

            byte[] dadosBytes = Encoding.UTF8.GetBytes(jsonString);
            fluxoRede.Write(dadosBytes, 0, dadosBytes.Length);
            fluxoRede.Flush();
        }
        catch (Exception e)
        {
            Logger.LogError($"Erro ao transmitir JSON: {e.Message}");
            fluxoRede = null;
            clienteTcp = null;
        }
    }

    private void OnApplicationQuit()
    {
        if (clienteTcp != null) 
        {
            clienteTcp.Close();
        }
    }
}