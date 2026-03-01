# =============================================================
# main.py
# Punto de entrada principal del proyecto.
# Orquesta la carga del laberinto, la construcción del grafo,
# la ejecución de los algoritmos de búsqueda y la visualización
# de los resultados.
# =============================================================

from maze import Maze
from graph import matrix_to_graph
from search import dfs, bfs, a_star
from heuristics import build_heuristic
from visualization import draw_graph, draw_matrix_with_path
from macro import (
    identificar_nodos_decision,
    construir_macro_grafo,
    astar_macro,
    reconstruir_ruta_completa,
    build_heuristic_macro,
)
import ast
import time

# Laberinto de ejemplo (4x4): inicio 2, meta 3, paredes 1, libre 0.
# Úsese si al pedir la matriz se deja la entrada en blanco y se pulsa Enter.
LABERINTO_EJEMPLO = [
    [2, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [1, 0, 1, 3],
]


def leer_laberinto():
    print("Pegue la matriz del laberinto en formato Python.")
    print("Ejemplo: [[2,1,1],[0,0,1],[1,3,0]]")
    print("(Deje en blanco y Enter para usar el laberinto de ejemplo.)\n")

    entrada = input("Ingrese la matriz: ").strip()
    if not entrada:
        print("Usando laberinto de ejemplo.")
        return LABERINTO_EJEMPLO, len(LABERINTO_EJEMPLO)

    try:
        matrix = ast.literal_eval(entrada)

        # Validar que sea lista de listas
        if not isinstance(matrix, list) or not all(isinstance(fila, list) for fila in matrix):
            print("Error: Formato inválido.")
            return None, None

        N = len(matrix)

        # Validar que sea NxN
        for fila in matrix:
            if len(fila) != N:
                print("Error: La matriz no es NxN.")
                return None, None

        return matrix, N

    except:
        print("Error: Entrada inválida.")
        return None, None


def main():

    matrix, N = leer_laberinto()

    if matrix is None:
        return

    maze = Maze(matrix)

    start = maze.find_value(2)
    goal = maze.find_value(3)

    if start is None or goal is None:
        print("Error: No se encontró inicio (2) o meta (3).")
        return

    graph = matrix_to_graph(matrix)

    print("\nDimensión:", N)
    print("Inicio:", start)
    print("Meta:", goal)

    # ==============================
    # DFS
    # ==============================
    print("\n===== DFS =====")
    path_dfs, explored_dfs = dfs(graph, start, goal)

    print("Ruta:", path_dfs)
    print("Nodos explorados:", len(explored_dfs))
    print("Orden exploración:", explored_dfs)

    draw_graph(graph, path_dfs, explored_dfs)
    draw_matrix_with_path(matrix, path_dfs, explored_dfs)

    # ==============================
    # BFS
    # ==============================
    print("\n===== BFS =====")
    path_bfs, explored_bfs = bfs(graph, start, goal)

    print("Ruta:", path_bfs)
    print("Nodos explorados:", len(explored_bfs))
    print("Orden exploración:", explored_bfs)

    draw_graph(graph, path_bfs, explored_bfs)
    draw_matrix_with_path(matrix, path_bfs, explored_bfs)

    # ==============================
    # A* (original: celda a celda)
    # ==============================
    print("\n===== A* (original) =====")
    heuristic = build_heuristic(matrix, goal)
    t0 = time.perf_counter()
    path_astar, explored_astar = a_star(graph, start, goal, heuristic)
    tiempo_original = time.perf_counter() - t0
    nodos_expandidos_original = len(explored_astar)

    print("Ruta:", path_astar)
    print("Nodos explorados:", nodos_expandidos_original)
    print("Orden exploración:", explored_astar)

    draw_graph(graph, path_astar, explored_astar)
    draw_matrix_with_path(matrix, path_astar, explored_astar)

    # ==============================
    # Abstracción: nodos de decisión y macro-grafo
    # ==============================
    nodos_decision = identificar_nodos_decision(matrix, inicio=start, meta=goal)
    macro_grafo = construir_macro_grafo(matrix, nodos_decision)
    heuristic_macro = build_heuristic_macro(nodos_decision, goal)

    # ==============================
    # A* macro (sobre nodos de decisión)
    # ==============================
    print("\n===== A* macro (optimizado) =====")
    t1 = time.perf_counter()
    path_astar_macro, explored_astar_macro, nodos_expandidos_macro = astar_macro(
        macro_grafo, start, goal, heuristic_macro
    )
    tiempo_macro = time.perf_counter() - t1
    print("Ruta (nodos de decisión):", path_astar_macro)
    print("Nodos expandidos:", nodos_expandidos_macro)
    print("Orden de expansión:", explored_astar_macro)

    if path_astar_macro:
        ruta_completa_macro = reconstruir_ruta_completa(
            path_astar_macro, matrix, nodos_decision
        )
        print("Ruta completa reconstruida:", ruta_completa_macro)
        draw_matrix_with_path(matrix, ruta_completa_macro, explored_astar_macro)
    else:
        print("No se encontró ruta con A* macro.")

    # ==============================
    # Análisis comparativo de rendimiento
    # ==============================
    print("\n===== Análisis comparativo =====")
    print("A* original:")
    print("  Nodos expandidos:", nodos_expandidos_original)
    print("  Tiempo (s): {:.6f}".format(tiempo_original))
    print("A* macro:")
    print("  Nodos expandidos:", nodos_expandidos_macro)
    print("  Tiempo (s): {:.6f}".format(tiempo_macro))
    if nodos_expandidos_original > 0:
        reduccion = (1 - nodos_expandidos_macro / nodos_expandidos_original) * 100
        print("Reducción porcentual (nodos expandidos): {:.1f}%".format(reduccion))


if __name__ == "__main__":
    main()
