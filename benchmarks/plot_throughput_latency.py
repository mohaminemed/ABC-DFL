import matplotlib.pyplot as plt
import numpy as np

# Data
functions = ['registerMP', 'registerCS', 'publishModel', 
             'joinCSModel', 'submitIM', 'submitGM', 'updateCSScores','distributeReward']

# Normal setting (original values)
max_throughput = [64.1, 51.4, 29.5, 61.8, 61.2, 62.2, 50.3,50.8]
max_latency = [1.65, 2.06, 3.54, 1.89, 1.73, 1.71, 2.13, 2.09]

# Byzantine setting (20%) - Arbitrary values
byzantine_throughput = [57.3, 47.5, 27.0, 58.1, 56.8, 56.3, 46.5, 47.0]  # Update as needed
byzantine_latency = [2.01, 2.81, 4.08, 2.72, 2.09, 2.08, 2.89, 2.84]  # Update as needed

color1 = "#7DB9FF"   
color2 = "#FF8080"
color3 = "#4CAF50"  # New color for Byzantine throughput
color4 = "#FFA500"  # New color for Byzantine latency

# Create figure and axis objects
fig, ax1 = plt.subplots(figsize=(13, 7.5))
plt.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.1)

# Plot for max throughput (normal)
ax1.set_ylabel('Max Throughput (tx/s)', fontsize=28)
bars = ax1.bar(functions, max_throughput, color=color1, hatch='\\', label='Max Throughput')

# Plot for Byzantine throughput
bars_b = ax1.bar(functions, byzantine_throughput, color=color3, hatch='//', alpha=0.7, label='Byzantine (20%) Throughput')

ax1.tick_params(axis='y', labelsize=20)
ax1.tick_params(axis='x', labelsize=20, rotation=20)

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom',  fontsize=16)

for bar in bars_b:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2, yval + 1, round(yval, 2), ha='center', va='bottom',  fontsize=16)

# Create second y-axis for max latency
ax2 = ax1.twinx()
ax2.set_ylabel('Max Latency (s)', fontsize=28)
ax2.plot(functions, max_latency, color=color2, marker='o', label='Max Latency', linewidth=5)
ax2.plot(functions, byzantine_latency, color=color4, marker='s', linestyle='dashed', linewidth=5, label='Byzantine (20%) Latency')
ax2.tick_params(axis='y', labelsize=20)

plt.grid(True)

# Add a legend with larger font size
ax1.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.25, 1.22))
ax2.legend(fontsize=20, loc="upper center", bbox_to_anchor=(0.75, 1.22))

fig.tight_layout()

# Save the figure
plt.savefig("throughput_byzantine.pdf", format="pdf")
# plt.show()  # Uncomment this line if you want to display the plot
