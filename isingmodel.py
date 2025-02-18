import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter 
from matplotlib.colors import ListedColormap

# Paleta de cores personalizada
colors = ['#FF0000', '#0000FF']  # Vermelho para -1 (↓), Azul para +1 (↑)
cmap = ListedColormap(colors)

def initialize_grid(n):
    """Inicializa a grade com spins aleatórios (-1 ou 1)."""
    return np.random.choice([-1, 1], size=(n, n))

def get_neighbors_sum(grid, i, j):
    """Calcula a soma dos spins vizinhos com condições de contorno periódicas."""
    n = grid.shape[0]
    up = grid[(i-1) % n, j]
    down = grid[(i+1) % n, j]
    left = grid[i, (j-1) % n]
    right = grid[i, (j+1) % n]
    return up + down + left + right

def metropolis_step(grid, beta):
    """Executa um passo do algoritmo de Metropolis-Hastings."""
    n = grid.shape[0]
    i, j = np.random.randint(0, n, size=2)
    current_spin = grid[i, j]
    neighbors_sum = get_neighbors_sum(grid, i, j)
    delta_E = 2 * current_spin * neighbors_sum
    
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        grid[i, j] *= -1
    return grid

def update(frame, img, grid, beta):
    """Atualiza a animação com novos spins."""
    for _ in range(100):  # Passos por frame
        grid = metropolis_step(grid, beta)
    img.set_data(grid)
    return img,

# Parâmetros
n = 150
beta = 2.5  # Valor intermediário para visualização interessante
grid = initialize_grid(n)

# Configuração da figura
fig, ax = plt.subplots(figsize=(8,8))
img = ax.imshow(grid, cmap=cmap, interpolation='none', vmin=-1, vmax=1)
ax.set_title(f'Modelo de Ising - β = {beta}\nVermelho: ↓ | Azul: ↑', fontsize=14)

# Adiciona barra de cores
cbar = fig.colorbar(img, ax=ax, ticks=[-1, 1])
cbar.ax.set_yticklabels(['Spin ↓ (-1)', 'Spin ↑ (+1)'], fontsize=12)
plt.axis('off')

# Animação
ani = FuncAnimation(fig, update, fargs=(img, grid, beta), frames=200, interval=50, blit=True)

plt.show()

ani.save("gif/ising_simu_b3.gif", writer=PillowWriter(fps=30))

