# Treinamento por etapas

O currículo é executado manualmente, uma etapa por vez. Cada etapa tem sua própria
função de recompensa, limite de episódio, arquivo de modelo, telemetria, monitor,
checkpoints e vídeos.

## Ordem

1. `etapas\Etapa01_Curar.py`: reforça pontos de vida efetivamente recuperados.
2. `etapas\Etapa02_Sobreviver.py`: recompensa permanecer vivo e pune dano recebido.
3. `etapas\Etapa03_DarDano.py`: reforça a redução da vida do chefe; o mod centraliza
   a Mãe Musgo na arena, desloca-a 215 pixels de tela abaixo do centro e
   interrompe seus comportamentos durante essa etapa.
4. `etapas\Etapa04_Vencer.py`: reforça dano, vitória e vida restante.

Os parâmetros de vida máxima do chefe e limite do episódio ficam no arquivo de cada
etapa. Se a modificação de um chefe alterar sua vida máxima, atualize
`HP_CHEFE_MAX` na etapa correspondente e `VIDA_MAXIMA_MOTHER` em
`SilksongRLTelemetriaMod.cs`; esses valores precisam coincidir.

## Execução

Na primeira etapa:

```powershell
python treinar_silksong.py --etapa curar
```

Depois de avaliar o resultado, escolha iniciar a próxima etapa:

```powershell
python treinar_silksong.py --etapa sobreviver
python treinar_silksong.py --etapa dano
python treinar_silksong.py --etapa vencer
```

Por padrão, uma etapa carrega o próprio modelo salvo para continuar; se ainda não
existir, carrega o modelo da etapa anterior. A etapa seguinte não inicia sem o
modelo anterior. Para informar outro modelo PPO compatível, use
`--modelo-inicial caminho\para\modelo.zip`. Para treinar mais ou menos passos,
use `--timesteps`; para alterar o limite por episódio, use `--limite-steps`.

Para iniciar um fluxo independente da etapa de dano (sem precisar treinar
Sobrevivência ou carregar um modelo de outra etapa), usando somente PPO:

```powershell
python testar_treino_dano.py
```

O script cria uma política PPO do zero, sem carregar ou executar o algoritmo
genético, e para após completar exatamente 25 partidas. Cada partida termina
por morte, derrota da Mãe Musgo (HP zero ou menor), limite de 5.000 steps ou
120 segundos do relógio do jogo. A morte da chefe encerra o episódio e solicita
o reset da arena. Quando a derrota da chefe é confirmada, o mod zera a vida da
Hornet e aguarda 5 segundos em tempo real antes de restaurar a arena.
No treino de dano, a FSM e a animação da chefe permanecem ativas enquanto a
física mantém seu movimento travado. Ao atingir 20 HP ou menos, mas ainda acima
de zero, o episódio termina sem bônus de vitória, e o reset restaura a vida da
chefe para 120. Se a chefe morrer antes de o reset ser processado, mantém-se o
fluxo de vitória existente.
É possível ajustar os limites por partida com `--limite-steps` e
`--limite-segundos`.
Morte e vitória encerram a partida e usam o reset da arena; a derrota não recebe
penalidade própria.

Na partida inicial, `testar_treino_dano.py` começa do estado atual do jogo sem
restaurar o HP da Hornet ou da Mãe Musgo. O mod aguarda 5 segundos em tempo real,
posiciona a chefe no centro da arena com deslocamento de 215 pixels para baixo e
a imobiliza. Nos resets seguintes, o mod a descongela antes de restaurar a arena
e espera 7 segundos em tempo real após o reset para imobilizá-la novamente.
O gráfico da recompensa por partida e sua média móvel fica em
`logs_treinamento\etapa_03_dar_dano\treino_ppo\desempenho_ppo.png`. O PPO e seus
checkpoints são salvos em `modelos_etapas\etapa_03_dar_dano\treino_ppo`. A
pontuação de cada episódio é exibida no terminal durante o treino e também no
resumo ao final. Se a lista do callback não estiver disponível no encerramento,
as pontuações são recuperadas de `monitor.csv` para imprimir o resumo e gerar o
gráfico.

A recompensa da etapa de dano combina o dano causado (escalado pela vida restante
da Hornet, como na V1), um custo de tempo de `0,01` por passo e uma penalidade de
1 ponto por vida perdida. Aproximar-se rende uma pequena recompensa proporcional
à redução da distância em relação ao passo anterior; afastar-se aplica a
penalidade correspondente. Ficar parado perto da chefe não gera recompensa de
proximidade. Derrotar a Mãe Musgo encerra a partida e concede bônus de vitória.
No fluxo PPO independente `testar_treino_dano.py`, o custo de tempo é zero:
ficar parado atacando não é penalizado, e cada dano efetivamente causado continua
rendendo recompensa. O tempo de jogo e o limite de steps ainda encerram partidas
sem progresso.
Esse fluxo também desconta 1 ponto ao cruzar cada limite lateral X da arena; a
penalidade só ocorre no cruzamento, não a cada passo passado da linha. O limite
inferior/chão não gera penalidade. O retângulo ciano tem recuo visual de 7 pixels
por borda; esse recuo não altera os limites de jogo usados pela penalidade.

## Observação da rede

O vetor atual tem 37 entradas:

- 16 valores de Hornet, Mãe Musgo, estado e limites da arena;
- 21 valores para os 7 alvos mais próximos (`delta_x`, `delta_y`, distância).

Inclui velocidade X/Y de ambos, HP da chefe, posição dela, um código de estado e
os quatro limites da arena. O código de estado é `0` indisponível, `1` parada,
`2` em movimento, `3` animação reconhecida como ataque, `4` derrotada. O nome de
animação exposto ao pesquisador vem do `Animator`; não é leitura direta da FSM
interna do jogo.

Os limites são valores iniciais configuráveis do BepInEx: X de `44.375` a `70.625`
e Y de `14.875` a `32.125`. A largura e a altura são 1,5 vez as medidas da caixa
anterior, mantendo o mesmo centro; ainda não representa os limites físicos
confirmados da arena.
O retângulo ciano é desenhado na tela com recuo lateral assimétrico: 64 pixels à
esquerda e 110 pixels à direita. Em relação aos valores anteriores, cada limite
lateral se aproxima mais 25 pixels do centro. Topo e base mantêm 7 pixels de
recuo cada.
Esse ajuste é apenas visual: os limites enviados à política permanecem iguais e
a caixa não impede a personagem de sair dela.

O mod identifica a Mãe Musgo pelo `HealthManager` e envia um único registro do
chefe, mesmo quando seus colliders aparecem como objetos distintos. A posição e
a velocidade do chefe são medidas pelo `Rigidbody2D` associado quando houver.

## Treinador genético separado (opcional)

O fluxo `testar_treino_dano.py` não usa este treinador genético. Ele permanece
disponível como ferramenta independente em `treinar_genetico.py`.

O algoritmo genético evolui a política de ação `[37, 64, 64, 7]`. Cada agente é
um conjunto de pesos e vieses; a pontuação do episódio na etapa de dano é sua
aptidão. A população tem 3 agentes por geração e executa 5 gerações (15
avaliações sequenciais no mesmo jogo). A cada geração, conserva o melhor agente,
cruza os dois melhores e aplica mutação Gaussiana com taxa de 1% e desvio 0,05.
Cada avaliação termina no primeiro limite atingido: 1.200 steps ou 120 segundos
do relógio de jogo. O tempo é medido pelo relógio escalado da Unity (`Time.time`),
portanto pausa do jogo não consome o limite. Se os 120 segundos forem atingidos,
o episódio é encerrado como derrota da Hornet e recebe a penalidade de morte da
etapa; o reset da arena usa o fluxo de limite de tempo existente. O teto de steps
continua como truncamento normal.

O melhor genoma é salvo em
`modelos_etapas\etapa_03_dar_dano\genetico\melhor_policy_genetica.npz`.
As pontuações individuais aparecem no console e em
`logs_treinamento\etapa_03_dar_dano\genetico\avaliacoes.csv`. O gráfico
`evolucao_genetica.png`, no mesmo diretório, compara a melhor, a média e a pior
pontuação de cada geração, além do melhor acumulado. Ele é atualizado ao fim de
cada geração concluída.

Treine o genético e depois use seus pesos como inicialização do PPO:

```powershell
python treinar_genetico.py
python treinar_silksong.py --etapa dano --modelo-genetico "modelos_etapas\etapa_03_dar_dano\genetico\melhor_policy_genetica.npz"
```

O PPO mantém o crítico de valor do modelo da etapa anterior, quando disponível,
e recebe os pesos genéticos no ator. O nome `transfer_learn` não é uma
dependência do projeto: a transferência de pesos é implementada localmente e
validada pelas dimensões do genoma.

## Arquivos gerados

- Modelos finais: `modelos_etapas\etapa_XX_nome.zip`
- Checkpoints: `modelos_etapas\checkpoints\etapa_XX_nome\`
- Métricas e telemetria: `logs_treinamento\etapa_XX_nome\`
- Vídeos: `videos_treinamento\etapa_XX_nome\`
- Avaliações genéticas: `logs_treinamento\etapa_03_dar_dano\genetico\`

As etapas não avançam automaticamente: o pesquisador avalia o modelo e decide
quando executar a próxima, para poder modificar o chefe ou o cenário entre fases.
