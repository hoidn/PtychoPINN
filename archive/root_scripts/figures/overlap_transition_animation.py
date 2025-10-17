import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Parameters
radius = 1.0
n_frames = 120  # Number of frames for smooth transition

# Define initial positions (90% overlap) and final positions (30% overlap)
initial_spacing = 0.45 * radius  # 90% overlap
final_spacing = 1.4 * radius     # 30% overlap

# Create initial grid positions for 90% overlap
initial_positions = []
for i in range(5):  # Reduced grid size to fit both states
    for j in range(4):
        x = i * initial_spacing
        y = j * initial_spacing
        initial_positions.append((x, y))

# Create final grid positions for 30% overlap
final_positions = []
for i in range(5):
    for j in range(4):
        x = i * final_spacing
        y = j * final_spacing
        final_positions.append((x, y))

# Convert to numpy arrays for easier interpolation
initial_positions = np.array(initial_positions)
final_positions = np.array(final_positions)

# Create circle patches (one for each position)
circles = []
for i in range(len(initial_positions)):
    circle = patches.Circle((0, 0), radius, 
                          facecolor='royalblue', 
                          edgecolor='darkblue',
                          alpha=0.4,
                          linewidth=2)
    ax.add_patch(circle)
    circles.append(circle)

# Set up the plot
ax.set_xlim(-1, 8)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_title('Ptychography Overlap Transition: 90% → 30%', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('X Position (a.u.)', fontsize=14)
ax.set_ylabel('Y Position (a.u.)', fontsize=14)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Text annotation that will be updated
text_annotation = ax.text(0.02, 0.98, '', 
                         transform=ax.transAxes, 
                         fontsize=12,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def animate(frame):
    # Calculate interpolation factor (0 to 1, then back to 0)
    if frame < n_frames // 2:
        t = frame / (n_frames // 2)  # 0 to 1
    else:
        t = (n_frames - frame) / (n_frames // 2)  # 1 to 0
    
    # Use smooth easing function
    t_smooth = 0.5 * (1 - np.cos(np.pi * t))
    
    # Interpolate positions
    current_positions = initial_positions + t_smooth * (final_positions - initial_positions)
    
    # Update circle positions
    for i, circle in enumerate(circles):
        circle.center = current_positions[i]
    
    # Update text annotation
    current_overlap = 90 - t_smooth * 60  # 90% to 30%
    current_spacing = initial_spacing + t_smooth * (final_spacing - initial_spacing)
    text_annotation.set_text(f'Overlap: {current_overlap:.1f}%\nSpacing: {current_spacing:.2f}r')
    
    return circles + [text_annotation]

# Create animation
anim = FuncAnimation(fig, animate, frames=n_frames, interval=100, blit=True, repeat=True)

# Save as GIF
print("Creating GIF animation... This may take a moment.")
anim.save('overlap_transition.gif', writer='pillow', fps=10, dpi=150)
plt.close()

print("Animation saved as overlap_transition.gif")