#!/usr/bin/env python3
"""
MPC Cost Landscape Visualisation — Enhanced
Convex vs Non-Convex with 3D and 2D contour views
PhD Preliminary Results: Hierarchical MPC-RL Navigation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# WORLD PARAMETERS
# ============================================================
goal        = np.array([0.0, -3.0])
robot_start = np.array([0.0,  2.5])

# U-Trap walls [x_min, x_max, y_min, y_max]
walls = [
    [-1.1, -0.9, 0.0, 5.0],
    [ 0.9,  1.1, 0.0, 5.0],
    [-1.1,  1.1, 4.8, 5.1],
]

# Grid
x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 7, 300)
X, Y = np.meshgrid(x, y)
Q = np.array([1.0, 1.0])

# ============================================================
# COST FUNCTIONS
# ============================================================
def convex_cost(X, Y, goal, Q):
    return Q[0]*(X - goal[0])**2 + Q[1]*(Y - goal[1])**2

def nonconvex_cost(X, Y, goal, Q, walls,
                   wall_penalty=80.0, wall_radius=0.25):
    cost = convex_cost(X, Y, goal, Q)
    for wall in walls:
        x_min, x_max, y_min, y_max = wall
        dx = np.maximum(0, np.maximum(x_min - X, X - x_max))
        dy = np.maximum(0, np.maximum(y_min - Y, Y - y_max))
        dist = np.sqrt(dx**2 + dy**2)
        cost += wall_penalty * np.exp(-dist / wall_radius)
    return cost

C1 = np.clip(convex_cost(X, Y, goal, Q),    0, 60)
C2 = np.clip(nonconvex_cost(X, Y, goal, Q, walls), 0, 60)

# ============================================================
# FIGURE
# ============================================================
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#0f0f0f')
fig.suptitle(
    'MPC Cost Landscape Analysis\n'
    'Convex (Obstacle-Free) vs Non-Convex (U-Trap) Environment\n'
    'PhD Preliminary Results — Hierarchical MPC-RL Navigation',
    fontsize=14, fontweight='bold', color='white', y=0.98)

# ============================================================
# PLOT 1 — 3D Convex
# ============================================================
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('#1a1a2e')
surf1 = ax1.plot_surface(X, Y, C1,
    cmap='cool', alpha=0.9,
    linewidth=0, antialiased=True)
ax1.scatter(goal[0], goal[1], 0,
    color='lime', s=120, zorder=10, label='Goal')
ax1.scatter(robot_start[0], robot_start[1],
    convex_cost(*robot_start, goal, Q),
    color='cyan', s=120, zorder=10, label='Robot Start')
ax1.set_title('3D Convex Cost Landscape\n(No Obstacles)',
    color='white', fontsize=11, pad=10)
ax1.set_xlabel('X (m)', color='white')
ax1.set_ylabel('Y (m)', color='white')
ax1.set_zlabel('Cost', color='white')
ax1.tick_params(colors='white')
ax1.view_init(elev=35, azim=-60)
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.4, pad=0.1)
cb1.ax.yaxis.set_tick_params(color='white')
plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')

# ============================================================
# PLOT 2 — 3D Non-Convex
# ============================================================
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_facecolor('#1a1a2e')
surf2 = ax2.plot_surface(X, Y, C2,
    cmap='plasma', alpha=0.9,
    linewidth=0, antialiased=True)
ax2.scatter(goal[0], goal[1], 0,
    color='lime', s=120, zorder=10, label='Goal')
ax2.scatter(robot_start[0], robot_start[1],
    nonconvex_cost(*robot_start, goal, Q, walls),
    color='cyan', s=120, zorder=10, label='Robot Start')
ax2.set_title('3D Non-Convex Cost Landscape\n(With U-Trap Obstacle)',
    color='white', fontsize=11, pad=10)
ax2.set_xlabel('X (m)', color='white')
ax2.set_ylabel('Y (m)', color='white')
ax2.set_zlabel('Cost', color='white')
ax2.tick_params(colors='white')
ax2.view_init(elev=35, azim=-60)
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.4, pad=0.1)
cb2.ax.yaxis.set_tick_params(color='white')
plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')

# ============================================================
# PLOT 3 — 2D Contour Convex
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#1a1a2e')
contour1 = ax3.contourf(X, Y, C1, levels=30, cmap='cool', alpha=0.9)
ax3.contour(X, Y, C1, levels=30,
    colors='white', alpha=0.2, linewidths=0.5)
ax3.scatter(*goal, color='lime', s=150, zorder=10,
    label='Goal', marker='*')
ax3.scatter(*robot_start, color='cyan', s=100, zorder=10,
    label='Robot Start', marker='o')
ax3.annotate('GOAL', goal, color='lime',
    fontsize=9, ha='left',
    xytext=(goal[0]+0.2, goal[1]+0.2))
ax3.annotate('ROBOT\nSTART', robot_start, color='cyan',
    fontsize=9, ha='left',
    xytext=(robot_start[0]+0.2, robot_start[1]+0.2))
ax3.set_title('2D Contour — Convex\n(Single Global Minimum)',
    color='white', fontsize=11)
ax3.set_xlabel('X (m)', color='white')
ax3.set_ylabel('Y (m)', color='white')
ax3.tick_params(colors='white')
ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)
cb3 = fig.colorbar(contour1, ax=ax3, shrink=0.8)
cb3.ax.yaxis.set_tick_params(color='white')
plt.setp(cb3.ax.yaxis.get_ticklabels(), color='white')
cb3.set_label('Cost', color='white')

# ============================================================
# PLOT 4 — 2D Contour Non-Convex with U-Trap drawn
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#1a1a2e')
contour2 = ax4.contourf(X, Y, C2, levels=30, cmap='plasma', alpha=0.9)
ax4.contour(X, Y, C2, levels=30,
    colors='white', alpha=0.2, linewidths=0.5)

# Draw U-trap walls on contour
for wall in walls:
    x_min, x_max, y_min, y_max = wall
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor='red',
        facecolor='darkred',
        alpha=0.8,
        label='_nolegend_'
    )
    ax4.add_patch(rect)

ax4.scatter(*goal, color='lime', s=150, zorder=10,
    label='Goal', marker='*')
ax4.scatter(*robot_start, color='cyan', s=100, zorder=10,
    label='Robot Start', marker='o')
ax4.annotate('GOAL', goal, color='lime',
    fontsize=9, ha='left',
    xytext=(goal[0]+0.2, goal[1]+0.2))
ax4.annotate('ROBOT\nSTART', robot_start, color='cyan',
    fontsize=9, ha='left',
    xytext=(robot_start[0]+0.2, robot_start[1]+0.2))

# Draw local minimum annotation inside trap
ax4.annotate('LOCAL\nMINIMUM\n(MPC gets\nstuck here)',
    xy=(0.0, 2.5), xytext=(-4.0, 3.5),
    color='yellow', fontsize=8,
    arrowprops=dict(arrowstyle='->', color='yellow'),
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#333333', edgecolor='yellow'))

# Draw global minimum annotation
ax4.annotate('GLOBAL\nMINIMUM\n(True Goal)',
    xy=(0.0, -3.0), xytext=(1.5, -3.5),
    color='lime', fontsize=8,
    arrowprops=dict(arrowstyle='->', color='lime'),
    bbox=dict(boxstyle='round,pad=0.3',
              facecolor='#333333', edgecolor='lime'))

ax4.set_title('2D Contour — Non-Convex\n(Local Minima + U-Trap Barrier)',
    color='white', fontsize=11)
ax4.set_xlabel('X (m)', color='white')
ax4.set_ylabel('Y (m)', color='white')
ax4.tick_params(colors='white')
ax4.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=9)

wall_patch = patches.Rectangle((0,0), 1, 1,
    facecolor='darkred', edgecolor='red', label='U-Trap Walls')
ax4.legend(handles=[
    plt.scatter([],[], color='lime', marker='*', label='Goal'),
    plt.scatter([],[], color='cyan', marker='o', label='Robot Start'),
    wall_patch],
    facecolor='#1a1a2e', labelcolor='white', fontsize=9)

cb4 = fig.colorbar(contour2, ax=ax4, shrink=0.8)
cb4.ax.yaxis.set_tick_params(color='white')
plt.setp(cb4.ax.yaxis.get_ticklabels(), color='white')
cb4.set_label('Cost', color='white')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
save_path = '/home/david/mpc_rl_robot_preliminary/results/screenshots/cost_landscape.png'
plt.savefig(save_path, dpi=150,
    bbox_inches='tight', facecolor='#0f0f0f')
print(f'[Saved] → {save_path}')
plt.show()