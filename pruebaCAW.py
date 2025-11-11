# eca.py - Autómata celular elemental (Wolfram)
# Requiere: python 3.10+, matplotlib, numpy
# Instalar si falta: pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from typing import List

def rule_number_to_lookup(rule: int) -> dict:
    """Convierte un número de regla (0-255) en un diccionario que mapea
    vecindarios (tuple de 3 bits) -> salida (0/1)."""
    if not (0 <= rule <= 255):
        raise ValueError("rule must be in 0..255")
    bits = [(rule >> i) & 1 for i in range(7, -1, -1)]  # 111..000
    patterns = [(1,1,1),(1,1,0),(1,0,1),(1,0,0),(0,1,1),(0,1,0),(0,0,1),(0,0,0)]
    return {patterns[i]: bits[i] for i in range(8)}

def step(state: np.ndarray, lookup: dict) -> np.ndarray:
    """Un paso del autómata con condiciones de borde periódicas (wrap)."""
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    new = np.zeros_like(state)
    for i in range(state.size):
        key = (int(left[i]), int(state[i]), int(right[i]))
        new[i] = lookup[key]
    return new

def run_eca(rule: int, size: int=201, steps: int=200, initial: str='single') -> np.ndarray:
    """Ejecuta el autómata y devuelve una matriz (steps+1, size) con la historia."""
    lookup = rule_number_to_lookup(rule)
    grid = np.zeros((steps+1, size), dtype=np.uint8)
    if initial == 'single':
        grid[0, size//2] = 1
    elif initial == 'random':
        rng = np.random.default_rng()
        grid[0] = rng.integers(0,2,size=size)
    else:
        raise ValueError("initial must be 'single' or 'random'")
    for t in range(steps):
        grid[t+1] = step(grid[t], lookup)
    return grid

def plot_eca(grid: np.ndarray, rule: int, filename: str | None = None):
    plt.figure(figsize=(8, 8 * grid.shape[0] / grid.shape[1]))
    plt.imshow(grid, interpolation='nearest', aspect='auto')
    plt.title(f"Elementary Cellular Automaton — Rule {rule}")
    plt.xlabel("Cell index")
    plt.ylabel("Time step")
    plt.gca().invert_yaxis()  # opcional: tiempo 0 arriba
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=200)
        print(f"Guardado en {filename}")
    else:
        plt.show()

if __name__ == "__main__":
    # Parámetros: cambia rule a 30, 90, 110, etc.
    rule = 54
    size = 401
    steps = 300
    grid = run_eca(rule, size=size, steps=steps, initial='single')
    plot_eca(grid, rule, filename=f"rule_{rule}.png")
