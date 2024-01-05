import matplotlib.pyplot as plt

# Define the size of the grid world
num_rows = 11
num_cols = 11

# Create a figure and axis for the grid world
fig, ax = plt.subplots(figsize=(10, 10))

# Create grid lines for the cells
for i in range(num_rows + 1):
    ax.axhline(i, color='black', lw=2)
for j in range(num_cols + 1):
    ax.axvline(j, color='black', lw=2)

# Remove axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])

# Optionally, add labels to cells
# You can use a loop to add text to each cell as needed
# For example, adding "A", "B", "C", ... in each cell
for i in range(num_rows):
    for j in range(num_cols):
        cell_text = f'({i}, {j})'  # Modify this text as needed
        ax.text(j + 0.5, i + 0.5, cell_text, fontsize=12, ha='center', va='center')

# Set the aspect ratio to equal to ensure square cells
ax.set_aspect('equal')

# Show the grid world
plt.show()
