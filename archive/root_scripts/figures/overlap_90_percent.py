import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Parameters for 90% overlap
radius = 1.0
# For 90% overlap, distance between centers should be ~0.45*radius
spacing = 0.45 * radius

# Create a denser grid of circles
positions = []
for i in range(12):
    for j in range(10):
        x = i * spacing
        y = j * spacing
        positions.append((x, y))

# Plot circles with semi-transparent blue
for x, y in positions:
    circle = patches.Circle((x, y), radius, 
                          facecolor='royalblue', 
                          edgecolor='darkblue',
                          alpha=0.25,  # Lower alpha due to more overlap
                          linewidth=1.5)
    ax.add_patch(circle)

# Add some aesthetic elements
ax.set_xlim(-1, 6.5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3, linestyle='--')

# Style the plot
ax.set_title('Ptychography Scan Pattern: ~90% Overlap', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('X Position (a.u.)', fontsize=14)
ax.set_ylabel('Y Position (a.u.)', fontsize=14)

# Add annotation
ax.text(0.02, 0.98, f'Mean Overlap: ~90%\nBeam Spacing: {spacing:.2f}r', 
        transform=ax.transAxes, 
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Remove top and right spines for cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('overlap_90_percent.png', dpi=300, bbox_inches='tight')
plt.savefig('overlap_90_percent.svg', format='svg', bbox_inches='tight')
plt.close()

print("90% overlap visualization saved as overlap_90_percent.png and .svg")