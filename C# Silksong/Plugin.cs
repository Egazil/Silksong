using BepInEx;
using UnityEngine;
using System.Collections.Generic;

[BepInPlugin("com.seuusuario.silksongcontains", "Mod Arena Contains Flexivel", "1.0.0")]
public class SilksongContainsMod : BaseUnityPlugin
{
    private Transform hornetTransform;
    private Collider2D hornetCollider;
    
    private int vidaHornet = 0;
    private bool podeFocar = false;
    private Vector2 posicaoHornet = Vector2.zero;

    private int vidaMother = -1;
    private HealthManager motherHealthManager;

    private struct AlvoInfo
    {
        public Collider2D collider;
        public string nome;
        public float distancia;
        public Color cor;
    }

    private List<AlvoInfo> alvosDetectados = new List<AlvoInfo>();
    private Texture2D texturaBorda;
    private GUIStyle estiloTexto;

    private void Awake()
    {
        Logger.LogInfo("Mod Arena Contains Flexivel carregado!");
        texturaBorda = new Texture2D(1, 1);
        texturaBorda.SetPixel(0, 0, Color.white);
        texturaBorda.Apply();

        estiloTexto = new GUIStyle();
        estiloTexto.normal.textColor = Color.white;
        estiloTexto.fontSize = 12;
    }

    private void Update()
    {
        if (hornetTransform == null || hornetCollider == null)
        {
            var hero = Object.FindFirstObjectByType<HeroController>();
            if (hero != null)
            {
                hornetTransform = hero.transform;
                hornetCollider = hornetTransform.GetComponentInChildren<Collider2D>();
            }
        }

        if (hornetTransform != null) posicaoHornet = hornetTransform.position;

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

        if (motherHealthManager != null) vidaMother = motherHealthManager.hp;
        else vidaMother = -1;

        EscanearComContains();
    }

    private void EscanearComContains()
    {
        alvosDetectados.Clear();
        var colisores = Object.FindObjectsByType<Collider2D>(FindObjectsSortMode.None);

        foreach (var col in colisores)
        {
            if (col == null || !col.gameObject.activeInHierarchy) continue;

            string nomeObj = col.gameObject.name;

            // Uso do Contains (ignorando maiúsculas/minúsculas via conversão para lowercase)
            string nomeLower = nomeObj.ToLower();

            bool ehStalactite = nomeLower.Contains("stalactite");
            bool ehMother = nomeLower.Contains("mother");
            bool ehCrawler = nomeLower.Contains("crawler");

            if (ehStalactite || ehMother || ehCrawler)
            {
                if (ehMother && motherHealthManager == null)
                {
                    motherHealthManager = col.GetComponentInChildren<HealthManager>() ?? col.GetComponentInParent<HealthManager>();
                }

                float dist = 0f;
                if (hornetCollider != null)
                {
                    dist = Vector2.Distance(hornetCollider.bounds.center, col.bounds.center);
                }

                // Cores solicitadas: Vermelho (Mãe), Amarelo (Crawler), Verde (Stalactite)
                Color corAlvo = Color.white;
                if (ehMother) corAlvo = Color.red;
                else if (ehCrawler) corAlvo = Color.yellow;
                else if (ehStalactite) corAlvo = Color.green;

                alvosDetectados.Add(new AlvoInfo
                {
                    collider = col,
                    nome = nomeObj,
                    distancia = dist,
                    cor = corAlvo
                });
            }
        }
    }

    private void OnGUI()
    {
        if (Camera.main != null)
        {
            if (hornetCollider != null) DesenharBoxComTexto(hornetCollider, "Hornet", Color.cyan);

            foreach (var alvo in alvosDetectados)
            {
                if (alvo.collider != null)
                {
                    string textoCaixa = $"{alvo.nome}\n{alvo.distancia.ToString("F1")} un";
                    DesenharBoxComTexto(alvo.collider, textoCaixa, alvo.cor);
                }
            }
        }

        int alturaPainel = 170 + (alvosDetectados.Count * 20);
        GUI.Box(new Rect(10, 10, 360, alturaPainel), "Painel Contains() - Silksong");
        GUI.Label(new Rect(20, 30, 340, 20), "Vida Hornet: " + vidaHornet);
        GUI.Label(new Rect(20, 50, 340, 20), "Vida Mãe Musgo: " + (vidaMother >= 0 ? vidaMother.ToString() : "Aguardando..."));
        GUI.Label(new Rect(20, 70, 340, 20), "Posição X: " + posicaoHornet.x.ToString("F1") + " | Y: " + posicaoHornet.y.ToString("F1"));
        GUI.Label(new Rect(20, 90, 340, 20), "Pode Focar: " + (podeFocar ? "SIM" : "Não"));
        GUI.Label(new Rect(20, 110, 340, 20), "--- Alvos Ativos: " + alvosDetectados.Count + " ---");

        int offsetPosY = 130;
        foreach (var alvo in alvosDetectados)
        {
            GUI.Label(new Rect(20, offsetPosY, 340, 20), $"- {alvo.nome}: {alvo.distancia.ToString("F2")} un");
            offsetPosY += 20;
        }
    }

    private void DesenharBoxComTexto(Collider2D col, string textoExibido, Color cor)
    {
        Bounds bounds = col.bounds;
        Vector3 screenMin = Camera.main.WorldToScreenPoint(new Vector3(bounds.min.x, bounds.max.y, bounds.min.z));
        Vector3 screenMax = Camera.main.WorldToScreenPoint(new Vector3(bounds.max.x, bounds.min.y, bounds.max.z));

        float x = screenMin.x;
        float y = Screen.height - screenMin.y;
        float largura = screenMax.x - screenMin.x;
        float altura = screenMin.y - screenMax.y;

        if (screenMin.z > 0 && largura > 0 && altura > 0)
        {
            Rect rect = new Rect(x, y, largura, altura);
            DesenharContornoRetangulo(rect, cor, 2f);
            GUI.Label(new Rect(x, y - 35, 250, 35), textoExibido, estiloTexto);
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
}