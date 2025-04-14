import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory for graphs
os.makedirs('graphs', exist_ok=True)

# Set style for all plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# 1. Detection accuracy vs. distance
distances = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
vehicle_accuracy = np.array([96.8, 95.3, 94.1, 92.7, 90.2, 86.5, 83.1, 80.4, 77.2, 75.0])
person_accuracy = np.array([97.2, 96.5, 94.8, 92.9, 89.6, 85.2, 81.7, 78.3, 74.5, 71.2])
traffic_light_accuracy = np.array([98.1, 97.3, 95.9, 93.8, 91.7, 88.2, 84.6, 80.1, 75.3, 72.0])

plt.figure()
plt.plot(distances, vehicle_accuracy, 'b-', linewidth=2, marker='o', label='Vehicles')
plt.plot(distances, person_accuracy, 'g-', linewidth=2, marker='s', label='Pedestrians')
plt.plot(distances, traffic_light_accuracy, 'r-', linewidth=2, marker='^', label='Traffic Lights')
plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Distance (meters)')
plt.ylabel('Detection Accuracy (%)')
plt.title('Detection Accuracy vs. Distance')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('graphs/accuracy_vs_distance.png', dpi=300)

# 2. Processing speed vs. number of objects
objects = np.array([5, 10, 15, 20, 25, 30, 35, 40])
fps = np.array([29.8, 27.2, 24.7, 21.3, 18.6, 16.2, 14.5, 12.8])

plt.figure()
plt.plot(objects, fps, 'b-', linewidth=2, marker='o')
plt.axhline(y=20, color='r', linestyle='--', alpha=0.7, label='Real-time threshold (20 FPS)')
plt.xlabel('Number of Objects per Frame')
plt.ylabel('Processing Speed (FPS)')
plt.title('Processing Speed vs. Number of Objects')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('graphs/speed_vs_objects.png', dpi=300)

# 3. Accuracy vs. lighting conditions
categories = ['Vehicles', 'Pedestrians', 'Traffic Lights', 'Traffic Light State']
daytime = np.array([94.2, 93.5, 96.1, 91.8])
nighttime = np.array([86.5, 85.2, 87.3, 82.1])
overcast = np.array([90.3, 89.7, 92.5, 86.4])

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots()
day_bars = ax.bar(x - width, daytime, width, label='Daytime', color='gold')
night_bars = ax.bar(x, nighttime, width, label='Nighttime', color='navy')
overcast_bars = ax.bar(x + width, overcast, width, label='Overcast', color='gray')

ax.set_xlabel('Detection Category')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Detection Accuracy vs. Lighting Conditions')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

add_labels(day_bars)
add_labels(night_bars)
add_labels(overcast_bars)

plt.tight_layout()
plt.savefig('graphs/accuracy_vs_lighting.png', dpi=300)

# 4. Traffic light state recognition vs. distance
distances = np.array([10, 20, 30, 40, 50, 60, 70])
red_accuracy = np.array([97.2, 95.8, 94.5, 91.2, 87.4, 83.6, 79.8])
yellow_accuracy = np.array([94.1, 92.3, 89.7, 85.2, 80.8, 76.1, 71.5])
green_accuracy = np.array([95.6, 93.4, 91.3, 87.5, 82.9, 78.2, 73.8])
overall_accuracy = np.array([95.7, 93.8, 91.8, 88.0, 83.7, 79.3, 75.0])

plt.figure()
plt.plot(distances, red_accuracy, 'r-', linewidth=2, marker='o', label='Red Light')
plt.plot(distances, yellow_accuracy, 'y-', linewidth=2, marker='s', label='Yellow Light')
plt.plot(distances, green_accuracy, 'g-', linewidth=2, marker='^', label='Green Light')
plt.plot(distances, overall_accuracy, 'b-', linewidth=2, marker='x', label='Overall')
plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Distance (meters)')
plt.ylabel('Recognition Accuracy (%)')
plt.title('Traffic Light State Recognition Accuracy vs. Distance')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('graphs/traffic_light_accuracy_vs_distance.png', dpi=300)

# 5. Confusion Matrix Visualization for Traffic Light State
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

# Create a confusion matrix
cm = np.array([
    [312, 14, 3],
    [21, 187, 9],
    [5, 18, 173]
])

# Normalize the confusion matrix
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Normalized Frequency', rotation=-90, va="bottom")

# Add labels
classes = ['Red', 'Yellow', 'Green']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Add title and axes labels
ax.set_title('Traffic Light State Recognition Confusion Matrix')
ax.set_ylabel('Actual Class')
ax.set_xlabel('Predicted Class')

# Add text annotations
thresh = cm_norm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        text = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
        ax.text(j, i, text,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig('graphs/traffic_light_confusion_matrix.png', dpi=300)

print("All performance graphs generated successfully in the 'graphs' directory.") 