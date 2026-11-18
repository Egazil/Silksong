using BepInEx;
using BepInEx.Configuration;
using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Text;
using System.Globalization;
using System.Reflection;

[BepInPlugin("com.seuusuario.silksongrltelemetria", "Mod RL Telemetria Silksong Periódico", "3.6.0")]
public class SilksongRLTelemetriaMod : BaseUnityPlugin
{
    private Transform hornetTransform;
    private Collider2D hornetCollider;
    private HeroController heroController;
    private Rigidbody2D hornetRigidbody;
    private Camera cameraPrincipal;

    private int vidaHornet = -1;
    private bool vidaHornetValida = false;
    private bool morteRealObservada = false;
    private bool podeFocar = false;
    private Vector2 posicaoHornet = Vector2.zero;

    private int vidaMother = -1;
    private bool vidaMotherValida = false;
    private bool vitoriaRealObservada = false;
    private HealthManager motherHealthManager;
    private Transform motherTransform;
    private Rigidbody2D motherRigidbody;
    private Animator motherAnimator;
    private string motherAnimationState = "indisponivel";
    private int motherStateCode;

    // --- SAVESTATE: Posições de Respawn Instantâneo ---
    private Vector2 spawnHornet = Vector2.zero;
    private Vector2 spawnBoss = Vector2.zero;
    private bool spawnRegistrado = false;
    private string etapaTreino = "";
    private bool chefeParadoNoTreino = false;
    private bool constraintsChefeSalvas;
    private RigidbodyConstraints2D constraintsChefeOriginais;

    // Conexão TCP com o Python
    private TcpClient clienteTcp;
    private NetworkStream fluxoRede;
    private float tempoTentativaConexao = 0f;
    private string bufferComandosPython = "";

    // Controle de Transmissão Periódica
    private float tempoDesdeUltimaTelemetria = 0f;
    private int contadorTelemetria = 0;
    private const float INTERVALO_TELEMETRIA = 0.1f;
    private const int INTERVALO_OBJETOS_DETALHADOS = 10;
    private const int VIDA_MAXIMA_MOTHER = 120;
    private ConfigEntry<float> arenaMinX;
    private ConfigEntry<float> arenaMaxX;
    private ConfigEntry<float> arenaMinY;
    private ConfigEntry<float> arenaMaxY;
    private bool telemetriaAtiva = false;
    private bool estaEmReset = false;
    private bool componentesChefeLogados = false;
    private bool erroCameraPosicaoChefeRegistrado = false;
    private const float RAIO_BUSCA_ALVOS = 100.0f;
    private const float DESLOCAMENTO_CHEFE_ABAIXO_CENTRO_PIXELS = 215f;
    private const float RECUO_BORDA_ESQUERDA_PIXELS = 64f;
    private const float RECUO_BORDA_DIREITA_PIXELS = 110f;
    private const float RECUO_BORDA_VERTICAL_PIXELS = 7f;
    private const int SEGMENTOS_CIRCULO_BUSCA = 64;

    // Bounding Box Visual para Depuração do Raio de Visão
    private Texture2D texturaBorda;
    private GUIStyle estiloTextoDebug;

    private void Awake()
    {
        Logger.LogInfo("Mod RL Telemetria (Transmissão Periódica + Savestate Instantâneo) iniciado!");

        arenaMinX = Config.Bind("Arena", "MinX", 44.375f, "Limite esquerdo inicial da arena.");
        arenaMaxX = Config.Bind("Arena", "MaxX", 70.625f, "Limite direito inicial da arena.");
        arenaMinY = Config.Bind("Arena", "MinY", 14.875f, "Limite inferior inicial da arena.");
        arenaMaxY = Config.Bind("Arena", "MaxY", 32.125f, "Limite superior inicial da arena.");
        if (arenaMinX.Value >= arenaMaxX.Value || arenaMinY.Value >= arenaMaxY.Value)
        {
            throw new InvalidOperationException(
                "Configuração inválida da arena: cada limite mínimo precisa ser menor que o máximo.");
        }
        
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

        AtualizarReferenciasJogo();
        ColetarEstadoAtual();

        // Inicia o fluxo apenas após o Python confirmar que está pronto para ler.
        if (telemetriaAtiva && !estaEmReset)
        {
            tempoDesdeUltimaTelemetria += Time.unscaledDeltaTime;
            if (tempoDesdeUltimaTelemetria >= INTERVALO_TELEMETRIA)
            {
                tempoDesdeUltimaTelemetria %= INTERVALO_TELEMETRIA;
                contadorTelemetria++;
                bool incluirObjetos = contadorTelemetria % INTERVALO_OBJETOS_DETALHADOS == 1;
                TransmitirJsonParaPython(incluirObjetosDetectados: incluirObjetos);
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
                    bufferComandosPython += Encoding.UTF8.GetString(bufferBytes, 0, bytesLidos);
                    while (bufferComandosPython.Contains("\n"))
                    {
                        int indiceNovaLinha = bufferComandosPython.IndexOf('\n');
                        string comando = bufferComandosPython.Substring(0, indiceNovaLinha).Trim();
                        bufferComandosPython = bufferComandosPython.Substring(indiceNovaLinha + 1);
                        if (!string.IsNullOrEmpty(comando))
                        {
                            ProcessarComandoPython(comando);
                        }
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

    private void AtualizarReferenciasJogo()
    {
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
            if (hornetRigidbody == null)
            {
                hornetRigidbody = ObterRigidbody2D(hornetTransform);
            }
        }

        if (motherHealthManager == null)
        {
            Collider2D[] alvos = Physics2D.OverlapCircleAll(posicaoHornet, RAIO_BUSCA_ALVOS);
            foreach (var col in alvos)
            {
                if (col == null || !col.gameObject.activeInHierarchy) continue;
                if (!col.gameObject.name.ToLower().Contains("mother")) continue;

                motherHealthManager =
                    col.GetComponentInChildren<HealthManager>() ??
                    col.GetComponentInParent<HealthManager>();
                if (motherHealthManager != null) break;
            }
        }

        AtualizarReferenciasChefe();
    }

    private Rigidbody2D ObterRigidbody2D(Transform origem)
    {
        if (origem == null) return null;
        return origem.GetComponent<Rigidbody2D>() ??
               origem.GetComponentInParent<Rigidbody2D>() ??
               origem.GetComponentInChildren<Rigidbody2D>(true);
    }

    private void AtualizarReferenciasChefe()
    {
        if (motherHealthManager == null)
        {
            motherTransform = null;
            motherRigidbody = null;
            motherAnimator = null;
            return;
        }

        motherTransform = motherHealthManager.transform;
        motherRigidbody = ObterRigidbody2D(motherTransform);
        motherAnimator = motherTransform.GetComponentInChildren<Animator>(true);
        AtualizarEstadoChefe();
    }

    private Vector2 ObterPosicaoChefe()
    {
        if (motherRigidbody != null) return motherRigidbody.position;
        return motherTransform != null ? (Vector2)motherTransform.position : Vector2.zero;
    }

    private Vector2 ObterVelocidadeChefe()
    {
        return motherRigidbody != null ? motherRigidbody.linearVelocity : Vector2.zero;
    }

    private string ObterNomeAnimacaoChefe()
    {
        if (motherAnimator == null || !motherAnimator.isActiveAndEnabled)
        {
            return "sem_animator_ativo";
        }

        AnimatorClipInfo[] clips = motherAnimator.GetCurrentAnimatorClipInfo(0);
        if (clips.Length > 0 && clips[0].clip != null)
        {
            return clips[0].clip.name;
        }

        return "state_" + motherAnimator.GetCurrentAnimatorStateInfo(0).shortNameHash;
    }

    private void AtualizarEstadoChefe()
    {
        if (motherHealthManager == null || !motherHealthManager.gameObject.activeInHierarchy)
        {
            motherStateCode = 0;
            motherAnimationState = "indisponivel";
            return;
        }

        if (motherHealthManager.hp <= 0)
        {
            motherStateCode = 4;
            motherAnimationState = "derrotada";
            return;
        }

        motherAnimationState = ObterNomeAnimacaoChefe();
        if (chefeParadoNoTreino)
        {
            motherStateCode = 1;
            motherAnimationState = "parada_treino";
            return;
        }

        string nome = motherAnimationState.ToLowerInvariant();
        if (nome.Contains("attack") || nome.Contains("spit") ||
            nome.Contains("slam") || nome.Contains("jump") ||
            nome.Contains("bite") || nome.Contains("hit"))
        {
            motherStateCode = 3;
        }
        else if (ObterVelocidadeChefe().sqrMagnitude > 0.01f)
        {
            motherStateCode = 2;
        }
        else
        {
            motherStateCode = 1;
        }
    }

    private HealthManager ObterHealthManager(Collider2D colisor)
    {
        return colisor.GetComponent<HealthManager>() ??
               colisor.GetComponentInParent<HealthManager>() ??
               colisor.GetComponentInChildren<HealthManager>();
    }

    private void ConfigurarEtapaTreino(string etapa)
    {
        if (string.IsNullOrWhiteSpace(etapa)) return;
        etapaTreino = etapa;
        bool pararChefe = etapa == "etapa_03_dar_dano";
        if (pararChefe)
        {
            AplicarModoChefeParado();
        }
        else
        {
            RestaurarComportamentoChefe();
        }
    }

    private void AplicarModoChefeParado()
    {
        if (motherTransform == null) return;

        chefeParadoNoTreino = true;
        if (motherRigidbody != null)
        {
            if (!constraintsChefeSalvas)
            {
                constraintsChefeOriginais = motherRigidbody.constraints;
                constraintsChefeSalvas = true;
            }
            motherRigidbody.constraints = RigidbodyConstraints2D.FreezeAll;
        }

        PosicionarChefeNoCentroDaArena();
        AtualizarEstadoChefe();
        Logger.LogInfo(
            "[Treino] FSM da Mãe Musgo ativa; movimento travado no centro da arena, " +
            "215 pixels abaixo do centro, com HP ativo.");
    }

    private void RestaurarComportamentoChefe()
    {
        chefeParadoNoTreino = false;
        if (motherRigidbody != null && constraintsChefeSalvas)
        {
            motherRigidbody.constraints = constraintsChefeOriginais;
            constraintsChefeSalvas = false;
        }
        AtualizarEstadoChefe();
    }

    private Vector2 CentroArena()
    {
        return new Vector2(
            (arenaMinX.Value + arenaMaxX.Value) * 0.5f,
            (arenaMinY.Value + arenaMaxY.Value) * 0.5f);
    }

    private void PosicionarChefeNoCentroDaArena()
    {
        if (!chefeParadoNoTreino || motherTransform == null) return;

        if (!TentarObterPosicaoChefeTreino(out Vector2 posicaoChefe)) return;
        if (motherRigidbody != null)
        {
            motherRigidbody.position = posicaoChefe;
            motherRigidbody.linearVelocity = Vector2.zero;
            motherRigidbody.angularVelocity = 0f;
        }
        else
        {
            motherTransform.position = new Vector3(
                posicaoChefe.x,
                posicaoChefe.y,
                motherTransform.position.z);
        }
    }

    private bool TentarObterPosicaoChefeTreino(out Vector2 posicaoChefe)
    {
        posicaoChefe = Vector2.zero;
        if (motherTransform == null) return false;

        if (cameraPrincipal == null || Screen.height <= 0)
        {
            if (!erroCameraPosicaoChefeRegistrado)
            {
                Logger.LogError(
                    "[Treino] Não foi possível posicionar a chefe 215 pixels abaixo do centro da arena: " +
                    "câmera principal ou altura da tela indisponível.");
                erroCameraPosicaoChefeRegistrado = true;
            }
            return false;
        }

        Vector2 centro = CentroArena();
        Vector3 centroMundo = new Vector3(
            centro.x,
            centro.y,
            motherTransform.position.z);
        Vector3 centroTela = cameraPrincipal.WorldToScreenPoint(centroMundo);
        if (centroTela.z <= 0f ||
            float.IsNaN(centroTela.x) || float.IsInfinity(centroTela.x) ||
            float.IsNaN(centroTela.y) || float.IsInfinity(centroTela.y))
        {
            if (!erroCameraPosicaoChefeRegistrado)
            {
                Logger.LogError(
                    "[Treino] O centro configurado da arena não está visível diante da câmera; " +
                    "não foi possível converter o deslocamento de 215 pixels para posição de mundo.");
                erroCameraPosicaoChefeRegistrado = true;
            }
            return false;
        }

        Vector3 posicaoAbaixoDoCentro = cameraPrincipal.ScreenToWorldPoint(
            new Vector3(
                centroTela.x,
                centroTela.y - DESLOCAMENTO_CHEFE_ABAIXO_CENTRO_PIXELS,
                centroTela.z));
        if (float.IsNaN(posicaoAbaixoDoCentro.y) ||
            float.IsInfinity(posicaoAbaixoDoCentro.y))
        {
            if (!erroCameraPosicaoChefeRegistrado)
            {
                Logger.LogError(
                    "[Treino] A conversão dos 215 pixels abaixo do centro da arena " +
                    "produziu uma posição inválida.");
                erroCameraPosicaoChefeRegistrado = true;
            }
            return false;
        }

        erroCameraPosicaoChefeRegistrado = false;
        posicaoChefe = new Vector2(centro.x, posicaoAbaixoDoCentro.y);
        return true;
    }

    private void FixedUpdate()
    {
        if (!chefeParadoNoTreino) return;
        PosicionarChefeNoCentroDaArena();
    }

    private void ProcessarComandoPython(string comando)
    {
        string[] partes = comando.Split(':');
        int resetId;

        if (partes.Length == 3 &&
            partes[0] == "START" &&
            int.TryParse(partes[1], out resetId))
        {
            IniciarSessaoSemReset(resetId, partes[2]);
            return;
        }

        if ((partes.Length == 2 || partes.Length == 3) &&
            partes[0] == "INIT" &&
            int.TryParse(partes[1], out resetId))
        {
            if (partes.Length == 3)
            {
                ConfigurarEtapaTreino(partes[2]);
            }
            if (!estaEmReset)
            {
                StartCoroutine(RotinaResetInstantaneo(resetId, "INIT"));
            }
            return;
        }

        if ((partes.Length == 3 || partes.Length == 4) &&
            partes[0] == "RESET" &&
            int.TryParse(partes[1], out resetId) &&
            (partes[2] == "MORTE" ||
             partes[2] == "VITORIA" ||
             partes[2] == "LIMITE_TEMPO" ||
             partes[2] == "LIMITE_STEPS" ||
             partes[2] == "LIMITE_HP_CHEFE"))
        {
            if (partes.Length == 4)
            {
                ConfigurarEtapaTreino(partes[3]);
            }
            if (!estaEmReset)
            {
                StartCoroutine(RotinaResetInstantaneo(resetId, partes[2]));
            }
            return;
        }

        Logger.LogWarning($"Comando ignorado por formato inválido: {comando}");
    }

    private void IniciarSessaoSemReset(int resetId, string etapa)
    {
        if (estaEmReset)
        {
            Logger.LogWarning(
                $"[Treino] Início {resetId} ignorado porque um reset já está em andamento.");
            return;
        }

        StartCoroutine(RotinaInicioSessaoSemReset(resetId, etapa));
    }

    private IEnumerator RotinaInicioSessaoSemReset(int resetId, string etapa)
    {
        estaEmReset = true;
        tempoDesdeUltimaTelemetria = 0f;
        etapaTreino = etapa;
        Logger.LogInfo(
            "[Treino] Aguardando 5 segundos em tempo real antes de configurar a chefe.");
        yield return new WaitForSecondsRealtime(5.0f);

        AtualizarReferenciasJogo();
        ColetarEstadoAtual();

        if (hornetTransform == null || !vidaHornetValida || vidaHornet <= 0 ||
            motherHealthManager == null || !vidaMotherValida || vidaMother <= 0)
        {
            RejeitarReset(
                resetId,
                "Não foi possível iniciar sem reset: Hornet ou Mãe Musgo indisponível ou sem vida.");
            yield break;
        }

        RegistrarComponentesChefe();
        spawnHornet = hornetTransform.position;
        ConfigurarEtapaTreino(etapa);
        if (chefeParadoNoTreino)
        {
            PosicionarChefeNoCentroDaArena();
            spawnBoss = ObterPosicaoChefe();
        }
        else
        {
            spawnBoss = motherHealthManager.transform.position;
        }

        spawnRegistrado = true;
        morteRealObservada = false;
        vitoriaRealObservada = false;
        ColetarEstadoAtual();
        TransmitirJsonParaPython(resetId, true, incluirObjetosDetectados: true);
        telemetriaAtiva = true;
        Logger.LogInfo(
            $"[Treino] Sessão iniciada sem reset. Posições preservadas -> " +
            $"Hornet: {spawnHornet} | Chefe: {spawnBoss} | " +
            $"HP Hornet: {vidaHornet} | HP chefe: {vidaMother}.");
        estaEmReset = false;
    }

    private void RegistrarComponentesChefe()
    {
        if (componentesChefeLogados || motherTransform == null) return;

        componentesChefeLogados = true;
        MonoBehaviour[] componentes =
            motherTransform.GetComponentsInChildren<MonoBehaviour>(true);

        foreach (MonoBehaviour componente in componentes)
        {
            if (componente == null) continue;

            string nomeFsm = "";
            string estadoFsm = "";
            if (componente.GetType().Name == "PlayMakerFSM")
            {
                try
                {
                    Type tipo = componente.GetType();
                    PropertyInfo propriedadeNome = tipo.GetProperty("FsmName");
                    PropertyInfo propriedadeEstado = tipo.GetProperty("ActiveStateName");
                    nomeFsm = propriedadeNome?.GetValue(componente, null)?.ToString() ?? "";
                    estadoFsm = propriedadeEstado?.GetValue(componente, null)?.ToString() ?? "";
                }
                catch (TargetInvocationException ex)
                {
                    Logger.LogWarning(
                        $"[Treino] Não foi possível ler o estado da FSM " +
                        $"'{componente.gameObject.name}': {ex.InnerException?.Message ?? ex.Message}");
                }
            }

            Logger.LogInfo(
                $"[Treino][ComponenteChefe] objeto='{componente.gameObject.name}' " +
                $"tipo='{componente.GetType().FullName}' ativo={componente.enabled} " +
                $"fsm='{nomeFsm}' estado='{estadoFsm}'");
        }
    }

    private void ColetarEstadoAtual()
    {
        try
        {
            vidaHornet = PlayerData.instance.GetInt("health");
            int sedaAtual = PlayerData.instance.GetInt("silk");
            podeFocar = (sedaAtual > 0);
            vidaHornetValida = vidaHornet >= 0;
        }
        catch
        {
            vidaHornet = -1;
            podeFocar = false;
            vidaHornetValida = false;
        }

        if (vidaHornetValida && vidaHornet == 0)
        {
            morteRealObservada = true;
        }

        if (motherHealthManager != null)
        {
            int hpChefe = motherHealthManager.hp;
            vidaMother = Math.Max(0, hpChefe);
            vidaMotherValida = true;
            if (hpChefe <= 0)
            {
                vitoriaRealObservada = true;
            }
        }
        else
        {
            vidaMother = vitoriaRealObservada ? 0 : -1;
            vidaMotherValida = vitoriaRealObservada;
        }
    }

    private string EscaparTextoJson(string texto)
    {
        if (string.IsNullOrEmpty(texto)) return "";
        return texto.Replace("\\", "\\\\")
                    .Replace("\"", "\\\"")
                    .Replace("\n", "\\n")
                    .Replace("\r", "\\r");
    }

    private void TransmitirJsonParaPython(
        int resetId = -1,
        bool resetCompleto = false,
        bool resetRejeitado = false,
        string motivoReset = "",
        bool incluirObjetosDetectados = false)
    {
        if (fluxoRede == null || clienteTcp == null || !clienteTcp.Connected) return;

        try
        {
            List<string> listaAlvosJson = new List<string>();
            List<string> listaObjetosJson = new List<string>();
            HashSet<int> alvosIncluidos = new HashSet<int>();
            Collider2D[] colisoresProximos = Physics2D.OverlapCircleAll(posicaoHornet, RAIO_BUSCA_ALVOS);

            foreach (var col in colisoresProximos)
            {
                if (col == null || !col.gameObject.activeInHierarchy) continue;

                string nomeObj = col.gameObject.name;
                string nomeLower = nomeObj.ToLower();
                bool ehMother = nomeLower.Contains("mother");
                bool ehAlvo = ehMother ||
                              nomeLower.Contains("stalactite") ||
                              nomeLower.Contains("crawler");
                if (!ehAlvo) continue;

                HealthManager healthManagerAlvo = null;
                int idObjeto;
                Vector2 centro;
                if (ehMother)
                {
                    healthManagerAlvo = ObterHealthManager(col);
                    if (healthManagerAlvo == null) continue;
                    if (motherHealthManager == null)
                    {
                        motherHealthManager = healthManagerAlvo;
                    }

                    idObjeto = healthManagerAlvo.GetInstanceID();
                    centro = healthManagerAlvo.transform.position;
                    nomeObj = healthManagerAlvo.gameObject.name;
                }
                else
                {
                    idObjeto = col.gameObject.GetInstanceID();
                    Vector3 centroBounds = col.bounds.center;
                    centro = new Vector2(centroBounds.x, centroBounds.y);
                }

                if (!alvosIncluidos.Add(idObjeto)) continue;

                Vector2 centroHornet = hornetCollider != null
                    ? (Vector2)hornetCollider.bounds.center
                    : posicaoHornet;
                float dist = (float)Math.Round(Vector2.Distance(centroHornet, centro), 2);
                float posX = (float)Math.Round(centro.x, 2);
                float posY = (float)Math.Round(centro.y, 2);

                string nomeEscapado = EscaparTextoJson(nomeObj);
                string posXStr = posX.ToString(CultureInfo.InvariantCulture);
                string posYStr = posY.ToString(CultureInfo.InvariantCulture);
                string distStr = dist.ToString(CultureInfo.InvariantCulture);
                string objetoJson =
                    $"{{\"nome\":\"{nomeEscapado}\",\"posX\":{posXStr},\"posY\":{posYStr},\"distancia\":{distStr}}}";

                if (incluirObjetosDetectados)
                {
                    listaObjetosJson.Add(objetoJson);
                }

                listaAlvosJson.Add(objetoJson);
            }

            string alvosArrayStr = "[" + string.Join(",", listaAlvosJson) + "]";
            string objetosArrayStr = incluirObjetosDetectados
                ? "[" + string.Join(",", listaObjetosJson) + "]"
                : "[]";

            if (motherHealthManager != null)
            {
                int hpChefe = motherHealthManager.hp;
                vidaMother = Math.Max(0, hpChefe);
                vidaMotherValida = true;
                if (hpChefe <= 0)
                {
                    vitoriaRealObservada = true;
                }
            }

            float hX = (float)Math.Round(posicaoHornet.x, 2);
            float hY = (float)Math.Round(posicaoHornet.y, 2);
            Vector2 hornetVelocity = hornetRigidbody != null
                ? hornetRigidbody.linearVelocity
                : Vector2.zero;
            Vector2 bossPosition = ObterPosicaoChefe();
            Vector2 bossVelocity = ObterVelocidadeChefe();

            string hXStr = hX.ToString(CultureInfo.InvariantCulture);
            string hYStr = hY.ToString(CultureInfo.InvariantCulture);
            string hVxStr = ((float)Math.Round(hornetVelocity.x, 3)).ToString(CultureInfo.InvariantCulture);
            string hVyStr = ((float)Math.Round(hornetVelocity.y, 3)).ToString(CultureInfo.InvariantCulture);
            string bossXStr = ((float)Math.Round(bossPosition.x, 2)).ToString(CultureInfo.InvariantCulture);
            string bossYStr = ((float)Math.Round(bossPosition.y, 2)).ToString(CultureInfo.InvariantCulture);
            string bossVxStr = ((float)Math.Round(bossVelocity.x, 3)).ToString(CultureInfo.InvariantCulture);
            string bossVyStr = ((float)Math.Round(bossVelocity.y, 3)).ToString(CultureInfo.InvariantCulture);
            string bossStateStr = motherStateCode.ToString(CultureInfo.InvariantCulture);
            string arenaMinXStr = arenaMinX.Value.ToString(CultureInfo.InvariantCulture);
            string arenaMaxXStr = arenaMaxX.Value.ToString(CultureInfo.InvariantCulture);
            string arenaMinYStr = arenaMinY.Value.ToString(CultureInfo.InvariantCulture);
            string arenaMaxYStr = arenaMaxY.Value.ToString(CultureInfo.InvariantCulture);
            string tempoJogoStr = Time.time.ToString(CultureInfo.InvariantCulture);
            string focarStr = podeFocar ? "true" : "false";
            string vidaHornetValidaStr = vidaHornetValida ? "true" : "false";
            string vidaMotherValidaStr = vidaMotherValida ? "true" : "false";
            string hornetDetectadaStr = hornetTransform != null ? "true" : "false";
            string mortoStr = vidaHornetValida && vidaHornet <= 0 ? "true" : "false";
            string morteRealStr = morteRealObservada ? "true" : "false";
            string vitoriaRealStr = vitoriaRealObservada ? "true" : "false";
            string resetCompletoStr = resetCompleto ? "true" : "false";
            string resetRejeitadoStr = resetRejeitado ? "true" : "false";
            string motivoResetJson = EscaparTextoJson(motivoReset);

            string jsonString =
                $"{{\"vida_hornet\":{vidaHornet},\"vida_hornet_valida\":{vidaHornetValidaStr}," +
                $"\"hornet_detectada\":{hornetDetectadaStr}," +
                $"\"pode_focar\":{focarStr},\"hornet_x\":{hXStr},\"hornet_y\":{hYStr}," +
                $"\"hornet_vx\":{hVxStr},\"hornet_vy\":{hVyStr}," +
                $"\"vida_mae\":{vidaMother},\"vida_mae_valida\":{vidaMotherValidaStr}," +
                $"\"mae_x\":{bossXStr},\"mae_y\":{bossYStr}," +
                $"\"mae_vx\":{bossVxStr},\"mae_vy\":{bossVyStr}," +
                $"\"mae_estado_codigo\":{bossStateStr}," +
                $"\"mae_estado_nome\":\"{EscaparTextoJson(motherAnimationState)}\"," +
                $"\"arena_min_x\":{arenaMinXStr},\"arena_max_x\":{arenaMaxXStr}," +
                $"\"arena_min_y\":{arenaMinYStr},\"arena_max_y\":{arenaMaxYStr}," +
                $"\"tempo_jogo\":{tempoJogoStr}," +
                $"\"chefe_detectada\":{vidaMotherValidaStr},\"morto\":{mortoStr}," +
                $"\"morte_real_confirmada\":{morteRealStr},\"vitoria_real_confirmada\":{vitoriaRealStr}," +
                $"\"alvos\":{alvosArrayStr},\"objetos_detectados\":{objetosArrayStr}," +
                $"\"objetos_detectados_atualizado\":{(incluirObjetosDetectados ? "true" : "false")}," +
                $"\"reset_id\":{resetId},\"reset_completo\":{resetCompletoStr}," +
                $"\"reset_rejeitado\":{resetRejeitadoStr},\"motivo_reset\":\"{motivoResetJson}\"}}\n";

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

    // --- VISUALIZAÇÃO DO ALCANCE DA VARREDURA ---
    private void OnGUI()
    {
        if (cameraPrincipal == null || hornetTransform == null) return;
        AtualizarEstadoChefe();

        Vector3 centroMundo = new Vector3(posicaoHornet.x, posicaoHornet.y, hornetTransform.position.z);
        Vector3 pontoAnteriorMundo = centroMundo + new Vector3(RAIO_BUSCA_ALVOS, 0f, 0f);
        Vector3 pontoAnteriorTela = cameraPrincipal.WorldToScreenPoint(pontoAnteriorMundo);

        for (int i = 1; i <= SEGMENTOS_CIRCULO_BUSCA; i++)
        {
            float angulo = 2f * Mathf.PI * i / SEGMENTOS_CIRCULO_BUSCA;
            Vector3 pontoAtualMundo = centroMundo + new Vector3(
                Mathf.Cos(angulo) * RAIO_BUSCA_ALVOS,
                Mathf.Sin(angulo) * RAIO_BUSCA_ALVOS,
                0f);
            Vector3 pontoAtualTela = cameraPrincipal.WorldToScreenPoint(pontoAtualMundo);

            if (pontoAnteriorTela.z > 0f && pontoAtualTela.z > 0f)
            {
                Vector2 inicio = new Vector2(pontoAnteriorTela.x, Screen.height - pontoAnteriorTela.y);
                Vector2 fim = new Vector2(pontoAtualTela.x, Screen.height - pontoAtualTela.y);
                DesenharLinhaTela(inicio, fim, Color.yellow, 2f);
            }

            pontoAnteriorTela = pontoAtualTela;
        }

        Vector3 topoTela = cameraPrincipal.WorldToScreenPoint(
            centroMundo + new Vector3(0f, RAIO_BUSCA_ALVOS, 0f));
        if (topoTela.z > 0f)
        {
            GUI.Label(
                new Rect(topoTela.x + 5f, Screen.height - topoTela.y, 250f, 20f),
                $"Alcance da varredura: raio {RAIO_BUSCA_ALVOS:F0} unidades",
                estiloTextoDebug);
        }

        DesenharLimitesArena();
        DesenharPainelEstadoChefe();
    }

    private void DesenharLimitesArena()
    {
        float z = hornetTransform.position.z;
        Vector3 inferiorEsquerdo = new Vector3(arenaMinX.Value, arenaMinY.Value, z);
        Vector3 inferiorDireito = new Vector3(arenaMaxX.Value, arenaMinY.Value, z);
        Vector3 superiorDireito = new Vector3(arenaMaxX.Value, arenaMaxY.Value, z);
        Vector3 superiorEsquerdo = new Vector3(arenaMinX.Value, arenaMaxY.Value, z);

        Vector3[] cantosTela =
        {
            cameraPrincipal.WorldToScreenPoint(inferiorEsquerdo),
            cameraPrincipal.WorldToScreenPoint(inferiorDireito),
            cameraPrincipal.WorldToScreenPoint(superiorDireito),
            cameraPrincipal.WorldToScreenPoint(superiorEsquerdo),
        };
        if (Array.Exists(cantosTela, canto => canto.z <= 0f)) return;

        float esquerda = float.PositiveInfinity;
        float direita = float.NegativeInfinity;
        float topo = float.PositiveInfinity;
        float inferior = float.NegativeInfinity;
        foreach (Vector3 canto in cantosTela)
        {
            esquerda = Mathf.Min(esquerda, canto.x);
            direita = Mathf.Max(direita, canto.x);
            topo = Mathf.Min(topo, Screen.height - canto.y);
            inferior = Mathf.Max(inferior, Screen.height - canto.y);
        }

        esquerda += RECUO_BORDA_ESQUERDA_PIXELS;
        direita -= RECUO_BORDA_DIREITA_PIXELS;
        topo += RECUO_BORDA_VERTICAL_PIXELS;
        inferior -= RECUO_BORDA_VERTICAL_PIXELS;
        if (direita <= esquerda || inferior <= topo) return;

        DesenharLinhaTela(
            new Vector2(esquerda, topo),
            new Vector2(direita, topo),
            Color.cyan,
            3f);
        DesenharLinhaTela(
            new Vector2(direita, topo),
            new Vector2(direita, inferior),
            Color.cyan,
            3f);
        DesenharLinhaTela(
            new Vector2(direita, inferior),
            new Vector2(esquerda, inferior),
            Color.cyan,
            3f);
        DesenharLinhaTela(
            new Vector2(esquerda, inferior),
            new Vector2(esquerda, topo),
            Color.cyan,
            3f);

        Vector3 rotulo = cameraPrincipal.WorldToScreenPoint(superiorEsquerdo);
        if (rotulo.z > 0f)
        {
            GUI.Label(
                new Rect(rotulo.x, Screen.height - rotulo.y - 22f, 420f, 22f),
                $"ARENA X[{arenaMinX.Value:F1}, {arenaMaxX.Value:F1}] " +
                $"Y[{arenaMinY.Value:F1}, {arenaMaxY.Value:F1}]",
                estiloTextoDebug);
        }
    }

    private void DesenharPainelEstadoChefe()
    {
        Vector2 velocidadeHornet = hornetRigidbody != null
            ? hornetRigidbody.linearVelocity
            : Vector2.zero;
        Vector2 velocidadeChefe = ObterVelocidadeChefe();
        Vector2 posicaoChefe = ObterPosicaoChefe();
        GUI.Box(new Rect(10f, 190f, 430f, 132f), "Telemetria para treinamento");
        GUI.Label(
            new Rect(20f, 214f, 410f, 20f),
            $"Hornet V=({velocidadeHornet.x:F2}, {velocidadeHornet.y:F2})");
        GUI.Label(
            new Rect(20f, 234f, 410f, 20f),
            $"Mãe Musgo HP={vidaMother} Pos=({posicaoChefe.x:F2}, {posicaoChefe.y:F2})");
        GUI.Label(
            new Rect(20f, 254f, 410f, 20f),
            $"Mãe Musgo V=({velocidadeChefe.x:F2}, {velocidadeChefe.y:F2}) " +
            $"Estado={motherStateCode}:{motherAnimationState}");
        GUI.Label(
            new Rect(20f, 274f, 410f, 20f),
            $"Etapa={etapaTreino} | Imobilizada={chefeParadoNoTreino}");
    }

    private void DesenharLinhaTela(Vector2 inicio, Vector2 fim, Color cor, float espessura)
    {
        Vector2 delta = fim - inicio;
        float comprimento = delta.magnitude;
        if (comprimento <= 0f) return;

        Vector2 centro = (inicio + fim) * 0.5f;
        Matrix4x4 matrizAnterior = GUI.matrix;
        Color corAntiga = GUI.color;

        try
        {
            GUIUtility.RotateAroundPivot(Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg, centro);
            GUI.color = cor;
            GUI.DrawTexture(
                new Rect(centro.x - comprimento * 0.5f, centro.y - espessura * 0.5f, comprimento, espessura),
                texturaBorda);
        }
        finally
        {
            GUI.matrix = matrizAnterior;
            GUI.color = corAntiga;
        }
    }

    private IEnumerator RotinaResetInstantaneo(int resetId, string motivo)
    {
        if (estaEmReset) yield break;

        estaEmReset = true;
        tempoDesdeUltimaTelemetria = 0f;

        if (motivo == "INIT")
        {
            Logger.LogInfo("[Savestate] Inicialização solicitada; aguardando 2 segundos.");
            yield return new WaitForSeconds(2.0f);

            AtualizarReferenciasJogo();
            ColetarEstadoAtual();
            if (hornetTransform == null || !vidaHornetValida ||
                motherHealthManager == null || !vidaMotherValida)
            {
                RejeitarReset(resetId, "Hornet ou chefe indisponível, ou vida inválida na inicialização.");
                yield break;
            }

            spawnHornet = hornetTransform.position;
            if (chefeParadoNoTreino)
            {
                PosicionarChefeNoCentroDaArena();
                spawnBoss = ObterPosicaoChefe();
            }
            else if (motherHealthManager != null)
            {
                spawnBoss = motherHealthManager.transform.position;
            }
            spawnRegistrado = true;

            try
            {
                PlayerData.instance.SetInt("health", 5);
                if (motherHealthManager != null) motherHealthManager.hp = VIDA_MAXIMA_MOTHER;
            }
            catch (Exception ex)
            {
                RejeitarReset(resetId, $"Falha ao inicializar vida do jogo: {ex.Message}");
                yield break;
            }

            AtualizarReferenciasJogo();
            ConfigurarEtapaTreino(etapaTreino);
            ColetarEstadoAtual();
            morteRealObservada = false;
            vitoriaRealObservada = false;
            Logger.LogInfo($"[Savestate] Posições salvas -> Hornet: {spawnHornet} | Chefe: {spawnBoss}.");
            TransmitirJsonParaPython(resetId, true, incluirObjetosDetectados: true);
            telemetriaAtiva = true;
            estaEmReset = false;
            yield break;
        }

        AtualizarReferenciasJogo();
        ColetarEstadoAtual();

        bool morteConfirmada = motivo == "MORTE" && morteRealObservada;
        bool vitoriaConfirmada = motivo == "VITORIA" && vitoriaRealObservada;
        bool limiteTempo = motivo == "LIMITE_TEMPO";
        bool limiteSteps = motivo == "LIMITE_STEPS";
        bool limiteHpChefe = motivo == "LIMITE_HP_CHEFE" &&
                             vidaMotherValida &&
                             vidaMother > 0 &&
                             vidaMother <= 20;
        if (!spawnRegistrado ||
            (!morteConfirmada && !vitoriaConfirmada && !limiteTempo &&
             !limiteSteps && !limiteHpChefe))
        {
            RejeitarReset(
                resetId,
                "O motivo do reset não corresponde a morte, vitória ou limite de episódio.");
            yield break;
        }

        bool reimobilizarChefe = chefeParadoNoTreino;

        if (vitoriaConfirmada)
        {
            try
            {
                PlayerData.instance.SetInt("health", 0);
                vidaHornet = 0;
                vidaHornetValida = true;
                morteRealObservada = true;
            }
            catch (Exception ex)
            {
                RejeitarReset(
                    resetId,
                    $"Falha ao zerar a vida da Hornet após a derrota da chefe: {ex.Message}");
                yield break;
            }

            Logger.LogInfo(
                "[Savestate] Mãe Musgo derrotada; Hornet eliminada. " +
                "Aguardando 5 segundos antes do reset.");
            yield return new WaitForSecondsRealtime(5.0f);
        }
        else if (limiteTempo || limiteSteps || limiteHpChefe)
        {
            string motivoLimite = limiteTempo
                ? "tempo"
                : limiteSteps ? "steps" : "HP da chefe";
            Logger.LogInfo(
                $"[Savestate] Reiniciando episódio encerrado pelo limite de {motivoLimite}.");
            yield return new WaitForSeconds(0.1f);
        }
        else
        {
            Logger.LogInfo("[Savestate] Aguardando 6 segundos antes de restaurar o estado...");
            yield return new WaitForSeconds(6.0f);
        }

        if (reimobilizarChefe)
        {
            RestaurarComportamentoChefe();
        }

        Logger.LogInfo("[Savestate] Executando respawn...");

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

        AtualizarReferenciasJogo();
        if (hornetTransform == null)
        {
            RejeitarReset(resetId, "Hornet não encontrada durante o respawn.");
            yield break;
        }

        hornetTransform.position = spawnHornet;
        var rb = hornetTransform.GetComponent<Rigidbody2D>();
        if (rb != null)
        {
            rb.linearVelocity = Vector2.zero;
        }

        try
        {
            PlayerData.instance.SetInt("health", 5);
            PlayerData.instance.SetInt("silk", 0);
        }
        catch (Exception ex)
        {
            RejeitarReset(resetId, $"Falha ao restaurar vida da Hornet: {ex.Message}");
            yield break;
        }

        if (motherHealthManager != null && motherHealthManager.gameObject != null)
        {
            var bossObj = motherHealthManager.gameObject;
            if (motherRigidbody != null)
            {
                motherRigidbody.position = spawnBoss;
                motherRigidbody.linearVelocity = Vector2.zero;
                motherRigidbody.angularVelocity = 0f;
            }
            else
            {
                bossObj.transform.position = spawnBoss;
            }
            motherHealthManager.hp = VIDA_MAXIMA_MOTHER;

            bossObj.SetActive(false);
            yield return null;
            bossObj.SetActive(true);
        }

        yield return null;

        if (reimobilizarChefe)
        {
            Logger.LogInfo(
                "[Savestate] Chefe descongelada após o reset; aguardando 7 segundos " +
                "antes de imobilizá-la novamente.");
            yield return new WaitForSecondsRealtime(7.0f);
        }

        AtualizarReferenciasJogo();
        if (reimobilizarChefe)
        {
            AplicarModoChefeParado();
        }
        ColetarEstadoAtual();
        if (!vidaHornetValida || vidaHornet <= 0 ||
            motherHealthManager == null || !vidaMotherValida)
        {
            RejeitarReset(resetId, "Respawn terminou sem Hornet e chefe válidas e ativas.");
            yield break;
        }

        morteRealObservada = false;
        vitoriaRealObservada = false;
        TransmitirJsonParaPython(resetId, true, incluirObjetosDetectados: true);
        estaEmReset = false;
        Logger.LogInfo("[Savestate] Respawn confirmado pelo mod.");
    }

    private void RejeitarReset(int resetId, string motivo)
    {
        Logger.LogError($"[Savestate] Reset {resetId} rejeitado: {motivo}");
        ColetarEstadoAtual();
        TransmitirJsonParaPython(resetId, false, true, motivo, incluirObjetosDetectados: true);
        estaEmReset = false;
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
        telemetriaAtiva = false;
    }

    private void OnApplicationQuit()
    {
        Desconectar();
    }
}
