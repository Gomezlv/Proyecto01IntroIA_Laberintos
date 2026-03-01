# =============================================================
# macro.py
# Abstracción del espacio de estados: nodos de decisión y macro-grafo.
# Permite que A* opere sobre intersecciones en lugar de celda a celda,
# reduciendo el número de nodos expandidos.
# =============================================================

from utils import reconstruct_path
from heuristics import manhattan
import heapq

# Direcciones de movimiento en la cuadrícula (arriba, abajo, izquierda, derecha)
DIRECCIONES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _grado_celda(laberinto, i, j):
    """
    Calcula el grado de una celda: número de vecinos válidos (transitables).

    Un vecino es válido si está dentro de los límites de la matriz y no es
    pared (valor != 1). Solo se considera para celdas transitables.
    """
    N = len(laberinto)
    if laberinto[i][j] == 1:  # Pared: no tiene grado en el grafo
        return 0
    grado = 0
    for di, dj in DIRECCIONES:
        ni, nj = i + di, j + dj
        if 0 <= ni < N and 0 <= nj < N and laberinto[ni][nj] != 1:
            grado += 1
    return grado


def identificar_nodos_decision(laberinto, inicio=None, meta=None):
    """
    Identifica los nodos de decisión del laberinto.

    Un nodo de decisión es cualquier celda transitable cuyo grado
    (número de vecinos válidos) sea distinto de 2. Esto incluye:
    - Inicio y meta (si se pasan, se añaden siempre)
    - Bifurcaciones (grado 3 o 4)
    - Callejones sin salida (grado 1)
    - Celdas en corredores tienen grado 2 y NO son nodos de decisión.

    Parámetros:
        laberinto: matriz NxN (lista de listas). 0=libre, 1=pared, 2=inicio, 3=meta.
        inicio: opcional, tupla (i, j) de la celda de inicio (siempre incluida).
        meta: opcional, tupla (i, j) de la celda meta (siempre incluida).

    Retorna:
        set de tuplas (i, j) con las coordenadas de los nodos de decisión.
    """
    N = len(laberinto)
    nodos = set()
    for i in range(N):
        for j in range(N):
            if laberinto[i][j] == 1:
                continue
            grado = _grado_celda(laberinto, i, j)
            if grado != 2:
                nodos.add((i, j))
    if inicio is not None:
        nodos.add(inicio)
    if meta is not None:
        nodos.add(meta)
    return nodos


def construir_macro_grafo(laberinto, nodos_decision):
    """
    Construye el macro-grafo donde los nodos son solo nodos de decisión
    y las aristas representan corredores completos entre ellos.

    Desde cada nodo de decisión se explora cada dirección posible,
    se avanza por el corredor hasta encontrar el siguiente nodo de decisión
    y se registra el peso como la longitud del corredor (número de pasos).

    Parámetros:
        laberinto: matriz NxN del laberinto.
        nodos_decision: set de coordenadas (i, j) de nodos de decisión.

    Retorna:
        dict: nodo_decision -> [(vecino_macro, costo), ...]
        donde costo = número de pasos (aristas) del corredor.
    """
    N = len(laberinto)
    macro_grafo = {nodo: [] for nodo in nodos_decision}

    for nodo in nodos_decision:
        ni, nj = nodo
        # Explorar cada dirección desde este nodo
        for di, dj in DIRECCIONES:
            vecino = (ni + di, nj + dj)
            vi, vj = vecino
            if vi < 0 or vi >= N or vj < 0 or vj >= N or laberinto[vi][vj] == 1:
                continue
            # Avanzar por el corredor hasta el siguiente nodo de decisión
            pasos = 1
            previo = nodo
            actual = vecino
            while actual not in nodos_decision:
                # En un corredor (grado 2) hay exactamente un "siguiente" (no el previo)
                candidatos = []
                for ddi, ddj in DIRECCIONES:
                    sig = (actual[0] + ddi, actual[1] + ddj)
                    if 0 <= sig[0] < N and 0 <= sig[1] < N and laberinto[sig[0]][sig[1]] != 1 and sig != previo:
                        candidatos.append(sig)
                if len(candidatos) != 1:
                    break  # Callejón sin salida o múltiples opciones (no debería en corredor)
                previo, actual = actual, candidatos[0]
                pasos += 1
            if actual in nodos_decision and actual != nodo:
                macro_grafo[nodo].append((actual, pasos))

    return macro_grafo


def astar_macro(macro_grafo, inicio, meta, heuristica):
    """
    A* sobre el macro-grafo: opera solo sobre nodos de decisión.
    Los vecinos son macro-vecinos (siguiente intersección) y g(n) acumula
    los pesos de las macro-aristas (longitud de los corredores).

    Parámetros:
        macro_grafo: dict nodo -> [(vecino, costo), ...]
        inicio: tupla (i, j) nodo de inicio (debe estar en macro_grafo).
        meta: tupla (i, j) nodo meta.
        heuristica: dict nodo -> h(n) (valor heurístico hacia meta), o callable(nodo)->float.

    Retorna:
        (ruta_macro, orden_expansion, num_nodos_expandidos)
        ruta_macro: lista de nodos de decisión desde inicio hasta meta, o None.
        orden_expansion: lista de nodos en orden de expansión.
        num_nodos_expandidos: len(orden_expansion).
    """
    if inicio not in macro_grafo or meta not in macro_grafo:
        return None, [], 0

    def h(n):
        return heuristica[n] if isinstance(heuristica, dict) else heuristica(n)

    open_set = []
    heapq.heappush(open_set, (0, inicio))
    g_cost = {inicio: 0}
    parent = {}
    closed = set()
    explored_order = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)
        explored_order.append(current)

        if current == meta:
            ruta = reconstruct_path(parent, inicio, meta)
            return ruta, explored_order, len(explored_order)

        for vecino, costo in macro_grafo.get(current, []):
            tentative_g = g_cost[current] + costo
            if vecino not in g_cost or tentative_g < g_cost[vecino]:
                parent[vecino] = current
                g_cost[vecino] = tentative_g
                f = tentative_g + h(vecino)
                heapq.heappush(open_set, (f, vecino))

    return None, explored_order, len(explored_order)


def _siguiente_decision_node(laberinto, nodos_decision, celda, desde):
    """
    Desde 'celda' (viniendo de 'desde'), camina por el corredor hasta
    alcanzar el siguiente nodo de decisión. Retorna ese nodo.
    """
    N = len(laberinto)
    previo = desde
    actual = celda
    while actual not in nodos_decision or actual == celda:
        vecinos = []
        for di, dj in DIRECCIONES:
            sig = (actual[0] + di, actual[1] + dj)
            if 0 <= sig[0] < N and 0 <= sig[1] < N and laberinto[sig[0]][sig[1]] != 1 and sig != previo:
                vecinos.append(sig)
        if len(vecinos) != 1:
            return actual if actual in nodos_decision else None
        previo, actual = actual, vecinos[0]
    return actual


def _obtener_celdas_corredor(laberinto, nodos_decision, desde, hasta):
    """
    Obtiene la secuencia de celdas al caminar desde 'desde' hasta 'hasta'
    por el corredor que los une. Retorna lista [celda1, celda2, ..., hasta]
    (sin repetir 'desde' al inicio).
    """
    N = len(laberinto)
    ruta = [desde]
    actual = desde
    previo = None
    while actual != hasta:
        vecinos = []
        for di, dj in DIRECCIONES:
            sig = (actual[0] + di, actual[1] + dj)
            if 0 <= sig[0] < N and 0 <= sig[1] < N and laberinto[sig[0]][sig[1]] != 1 and sig != previo:
                vecinos.append(sig)
        if not vecinos:
            return None
        if hasta in vecinos:
            ruta.append(hasta)
            return ruta[1:]  # excluir 'desde' para no duplicar al concatenar
        if len(vecinos) == 1:
            siguiente = vecinos[0]
        else:
            # Nodo de decisión: elegir el vecino cuyo corredor lleva a 'hasta'
            siguiente = None
            for v in vecinos:
                if _siguiente_decision_node(laberinto, nodos_decision, v, actual) == hasta:
                    siguiente = v
                    break
            if siguiente is None:
                siguiente = vecinos[0]
        ruta.append(siguiente)
        previo, actual = actual, siguiente
    return ruta[1:]


def reconstruir_ruta_completa(ruta_macro, laberinto, nodos_decision):
    """
    Expande la ruta compacta (solo nodos de decisión) en la lista completa
    de celdas desde el inicio hasta la meta, incluyendo todos los pasos
    intermedios en los corredores.

    Parámetros:
        ruta_macro: lista de nodos de decisión [(i1,j1), (i2,j2), ...] desde inicio a meta.
        laberinto: matriz NxN del laberinto.
        nodos_decision: set de coordenadas de nodos de decisión.

    Retorna:
        Lista de celdas (i, j) desde la primera hasta la última de ruta_macro,
        con todos los pasos intermedios en cada corredor.
    """
    if not ruta_macro:
        return []
    if len(ruta_macro) == 1:
        return list(ruta_macro)
    ruta_completa = [ruta_macro[0]]
    for k in range(1, len(ruta_macro)):
        desde, hasta = ruta_macro[k - 1], ruta_macro[k]
        celdas_corredor = _obtener_celdas_corredor(laberinto, nodos_decision, desde, hasta)
        if celdas_corredor is not None:
            ruta_completa.extend(celdas_corredor)
        else:
            ruta_completa.append(hasta)
    return ruta_completa


def build_heuristic_macro(nodos_decision, goal):
    """
    Construye el diccionario heurístico h(n) para todos los nodos de decisión,
    usando distancia Manhattan hasta la meta. Necesario para astar_macro.
    """
    return {nodo: manhattan(nodo, goal) for nodo in nodos_decision}
