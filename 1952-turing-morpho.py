import numpy as np  # Library for numerical operations and arrays
import matplotlib.pyplot as plt  # Library for creating visualizations
from matplotlib.animation import FuncAnimation  # Module for creating animations in Matplotlib
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# Grid size and parameters
N = 200  # Grid dimensions (N x N); larger N gives finer resolution but is computationally intensive
du, dv = 0.2, 0.1  # Diffusion rates: du for U (inhibitor, should be larger than dv for pattern formation), dv for V (activator)
f, k = 0.055, 0.062  # Feed rate (f) adds U over time; kill rate (k) removes V; specific values produce spots (try f=0.035, k=0.065 for stripes)
dt = 1.0  # Time step size for numerical stability (too large may cause instability)
steps = 5000  # Total number of simulation steps; more steps allow patterns to fully develop

# Initialize grids with random noise to break symmetry and initiate patterns
U = np.ones((N, N))  # U starts as uniform 1 (inhibitor everywhere)
V = np.zeros((N, N))  # V starts as uniform 0 (no activator initially)
# Seed a central square perturbation to kickstart pattern formation
U[N//2-20:N//2+20, N//2-20:N//2+20] = 0.5  # Lower U in center
V[N//2-20:N//2+20, N//2-20:N//2+20] = 0.25  # Add V in center
U += 0.05 * np.random.random((N, N))  # Add small random noise to U for realism
V += 0.05 * np.random.random((N, N))  # Add small random noise to V

# Laplacian function for diffusion: approximates second derivative using finite differences
# This models how chemicals spread to neighboring cells (up, down, left, right)
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +  # Vertical neighbors
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -  # Horizontal neighbors
            4 * Z)  # Subtract center cell four times

# Update function called for each animation frame
# Advances the simulation by applying reaction-diffusion equations
def update(frame):
    global U, V  # Use global variables to modify U and V in place
    for _ in range(10):  # Perform multiple sub-steps per frame to speed up simulation without slowing animation
        Lu = laplacian(U)  # Compute diffusion for U
        Lv = laplacian(V)  # Compute diffusion for V
        Uvv = U * V * V  # Reaction term: U inhibits V's growth quadratically
        # Update U: diffusion + feed - reaction
        U += dt * (du * Lu - Uvv + f * (1 - U))
        # Update V: diffusion + reaction - kill
        V += dt * (dv * Lv + Uvv - (f + k) * V)
    im.set_array(V)  # Update the image with current V values (activator patterns are visible here)
    return [im]  # Return the updated artist for blitting (efficient animation)

# Set up the figure and axis for plotting
fig, ax = plt.subplots(facecolor='black')  # Create a figure with a black background
ax.set_facecolor('black')

# Custom black-first colormap tuned to a neon bioelectric palette
# (deep blue base with cyan/green/yellow/orange highlights).
black_variation = LinearSegmentedColormap.from_list(
    "black_neon_bio",
    [
        (0.00, "#000000"),  # black background
        (0.10, "#040b2d"),  # deep navy
        (0.25, "#1e2f97"),  # electric blue
        (0.42, "#00b6ff"),  # cyan
        (0.58, "#00e676"),  # neon green
        (0.74, "#c6ff00"),  # yellow-green
        (0.88, "#ffd54f"),  # warm yellow
        (1.00, "#ff7043"),  # orange-red highlights
    ],
)

# PowerNorm brightens low-mid values to reveal more subtle structures.
im = ax.imshow(
    V,
    cmap=black_variation,
    norm=PowerNorm(gamma=0.75, vmin=0.0, vmax=0.65),
    interpolation='nearest',
)
ax.set_axis_off()
# Create animation: calls update for each frame, total frames = steps//10 to match sub-steps
ani = FuncAnimation(fig, update, frames=steps//10, interval=50, blit=True)  # interval=50ms between frames; blit optimizes redraw
plt.title('Turing Pattern Simulation (Gray-Scott Model)', color='white')  # Title for the plot
ani.save('turing_patterns_black_neon.gif', writer='pillow')  # Saves the neon variant as GIF
# plt.show()  # Optional: Still displays it

