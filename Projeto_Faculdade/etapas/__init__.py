from . import Etapa01_Curar, Etapa02_Sobreviver, Etapa03_DarDano, Etapa04_Vencer


ETAPAS = {
    "curar": Etapa01_Curar,
    "sobreviver": Etapa02_Sobreviver,
    "dano": Etapa03_DarDano,
    "vencer": Etapa04_Vencer,
}

ORDEM_ETAPAS = tuple(ETAPAS)


def obter_etapa(nome):
    try:
        return ETAPAS[nome]
    except KeyError as exc:
        opcoes = ", ".join(ETAPAS)
        raise ValueError(f"Etapa desconhecida: {nome!r}. Opções: {opcoes}.") from exc
