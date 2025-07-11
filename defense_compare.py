import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Example data
data = [
    [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # Comparing the Mean Distances(Threshold)(threshold = 3.4(mnist)/1.8(cifar)) Defense3
    [0.839, 0.897, 0.674, 0.764, 0.508, 0.619, 1.000, 1.000],  # Comparing Neighbors Defense2
    [1.000, 0.992, 1.000, 0.976, 1.000, 0.952, 1.000, 1.000],  # Clustering the Mean Distances Defense3 Siluhette(cluster = 3)
    [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],  # KNN Distance to Neighbor (threshold = 4) Defense1
]

# Rows (defenses)
rows = ['KNN Neighbor', 'Compare Neighbors', 'Cluster Distances', 'Compare Distances(tau)']

# Multi-level columns for hierarchical structure
columns = pd.MultiIndex.from_tuples([
    ('Precision', 'MNIST'), ('Precision', 'CIFAR'),
    ('Recall', 'MNIST'), ('Recall', 'CIFAR'),
    ('F1-Score', 'MNIST'), ('F1-Score', 'CIFAR'),
    ('Accuracy', 'MNIST'), ('Accuracy', 'CIFAR'),
])

# Create the DataFrame
df = pd.DataFrame(data, index=rows, columns=columns)

# Plot setup
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')  # Turn off axes for clean table view

# Table dimensions
nrows = len(rows) + 2  # Rows + 2 for hierarchical headers
ncols = len(columns)  # Total number of sub-columns
defense_width = 0.15  # Allocate less width for defense labels
cell_width = (1 - defense_width) / ncols
cell_height = 1 / nrows
sub_header_height = cell_height * 0.7  # Shorten height for MNIST/CIFAR

# Draw top-level headers
for i, top_col in enumerate(columns.levels[0]):
    x_start = defense_width + i * 2 * cell_width  # Adjust for defense width
    ax.add_patch(Rectangle((x_start, 1 - cell_height), 2 * cell_width, cell_height, facecolor="lightgray", edgecolor="black"))
    ax.text(x_start + cell_width, 1 - cell_height / 2, top_col, ha='center', va='center', fontsize=10)

# Draw sub-level headers (MNIST, CIFAR)
for i, (top_col, sub_col) in enumerate(columns):
    x_start = defense_width + i * cell_width
    ax.add_patch(Rectangle((x_start, 1 - cell_height - sub_header_height), cell_width, sub_header_height, facecolor="white", edgecolor="black"))
    ax.text(x_start + cell_width / 2, 1 - cell_height - sub_header_height / 2, sub_col, ha='center', va='center', fontsize=9)

# Draw row labels (Defense names)
for i, row_label in enumerate(rows):
    y_start = 1 - (i + 2) * cell_height - sub_header_height  # Start after headers
    ax.add_patch(Rectangle((0, y_start), defense_width, cell_height, facecolor="lightgray", edgecolor="black"))
    ax.text(defense_width / 2, y_start + cell_height / 2, row_label, ha='center', va='center', fontsize=10)

# Draw data cells
for i, row in enumerate(data):
    y_start = 1 - (i + 2) * cell_height - sub_header_height  # Start after headers
    for j, value in enumerate(row):
        x_start = defense_width + j * cell_width
        ax.add_patch(Rectangle((x_start, y_start), cell_width, cell_height, facecolor="white", edgecolor="black"))
        ax.text(x_start + cell_width / 2, y_start + cell_height / 2, f"{value:.3f}", ha='center', va='center', fontsize=10)

# Save and show the table
plt.savefig("shorter_sub_headers_table.png", dpi=300, bbox_inches='tight')
plt.show()
