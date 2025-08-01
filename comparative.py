import numpy as np
import matplotlib.pyplot as plt

#Accuracy
# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
accuracy_values = {
    "CNN": [46.15, 61.71, 71.12, 76.13, 80.79],
    "DCNN": [56.41, 50.45, 44.96, 48.73, 50.00],
    "DCNN + Attention": [51.28, 57.66, 57.77, 54.21, 54.42],
    "DCNN + Bi-LSTM + Attention": [53.99, 73.32, 87.63, 90.25, 92.88],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [71.4, 89, 92.91, 96.7, 99.45]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(accuracy_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Accuracy (%)", fontsize=22)
ax.set_title("Comparative Analysis of Accuracy", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
#plt.show()




#loss
# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]

loss_values = {
    "CNN": [15.42, 12.59, 11.01, 16.73, 29.65],
    "DCNN": [2.24, 3.6502, 1.73, 2.037, 2.132],
    "DCNN + Attention": [69.53, 68.36, 68.25, 69.04, 68.95],
    "DCNN + Bi-LSTM + Attention": [24.4, 23.34, 16.38, 12.29, 28.77],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [8.83, 9.07, 7.45, 7.03, 2.78]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(loss_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Loss (%)", fontsize=22)
ax.set_title("Comparative Analysis of Loss", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 80)
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)


# Show the plot
#plt.show()



#dice coefficient

# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
dice_values = {
    "CNN": [65.56, 68.12, 68.50, 67.67, 68.99],
    "DCNN": [71.90, 68.64, 72.78, 69.47, 70.42],
    "DCNN + Attention": [66.38, 67.80, 70.56, 68.23, 69.10],
    "DCNN + Bi-LSTM + Attention": [86.30, 92.36, 93.17, 91.98, 92.74],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [95.24, 96.47, 95.24, 91.30, 96.79]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(dice_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Dice Coefficient (%)", fontsize=22)
ax.set_title("Comparative Analysis of Dice Coefficient", fontsize=24)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
#plt.show()


#IOU

# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
iou_values = {
    "CNN": [47.35, 50.12, 48.63, 50.84, 46.68],
    "DCNN": [26.13, 23.39, 26.61, 23.87, 26.13],
    "DCNN + Attention": [24.84, 30.89, 26.77, 36.20, 29.11],
    "DCNN + Bi-LSTM + Attention": [89.02, 86.73, 82.33, 79.09, 74.07],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [90.10, 91.88, 88.21, 79.19, 91.27]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(iou_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Intersection Over Union (IOU) (%)", fontsize=22)
ax.set_title("Comparative Analysis of Intersection Over Union (IOU)", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)  # Set y-axis from 0 to 100
ax.set_yticks(np.arange(0, 101, 20))  # Set y-axis ticks from 0 to 100 with a step of 20
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
#plt.show()



# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
f1_values = {
    "CNN": [70.8, 74.54, 75.8, 76.77, 77.11],
    "DCNN": [30.16, 36.61, 28.78, 30.16, 38.09],
    "DCNN + Attention": [32.62, 38.47, 50.81, 56.54, 46.03],
    "DCNN + Bi-LSTM + Attention": [68.66, 87.39, 90.63, 88.6, 88.01],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [94.54, 95.76, 95.15, 95.76, 96.67]   			

}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(f1_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("F1-Score (%)", fontsize=22)
ax.set_title("Comparative Analysis of F1-Score", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 20))  
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
#plt.show()



# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
precision_values = {
    "CNN": [62.4, 67.66, 69.41, 69.7, 70.63],
    "DCNN": [62.42, 63.96, 64.51, 66.65, 67.52],
    "DCNN + Attention": [37.87, 49.44, 61.85, 32.2, 64.67],
    "DCNN + Bi-LSTM + Attention": [68.95, 70.31, 71.57, 86.4, 79.16],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [90.04, 94.35, 93.42, 94.35, 96.98]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(precision_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Precision (%)", fontsize=22)
ax.set_title("Comparative Analysis of Precision", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)  # Y-axis from 0 to 100
ax.set_yticks(np.arange(0, 101, 20))  # Y-axis ticks at intervals of 10
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
#plt.show()


#Recall
# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
recall_values = {
    "CNN": [54.29, 64.16, 65.93, 73.89, 74.25],
    "DCNN": [48.92, 50.77, 51.14, 49.79, 51.57],
    "DCNN + Attention": [49.06, 49.36, 49.36, 50.26, 50.85],
    "DCNN + Bi-LSTM + Attention": [62.45, 76.9, 84.32, 80.7, 87.64],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [92.74, 84.3, 86.32, 92.74, 96.97]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(recall_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Recall (%)", fontsize=22)
ax.set_title("Comparative Analysis of Recall", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)  # Y-axis from 0 to 100
ax.set_yticks(np.arange(0, 101, 20))  # Y-axis ticks at intervals of 20
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
plt.show()


#Sensitivity

# Define the data
x_labels = ["40%", "50%", "60%", "70%", "80%"]
sensitivity_values = {
    "CNN": [50.89, 56.21, 68.64, 73.96, 75.74],
    "DCNN": [51.4, 62.31, 72.4, 68.72, 78.21],
    "DCNN + Attention": [40.27, 29.51, 64.02, 91.6, 87.36],
    "DCNN + Bi-LSTM + Attention": [79.24, 89.96, 85.03, 92.08, 93.64],
    "Proposed DCNN + Bi-LSTM + Customised Mixed Attention Mech.": [94.01, 89.22, 91.62, 90.42, 97.47]
}
colors = ['skyblue', 'red', 'orange', 'lavender', 'lightgreen']

# Prepare x-axis positions
x = np.arange(len(x_labels))
width = 0.15  # Bar width

# Create plot
fig, ax = plt.subplots(figsize=(13, 7))

# Plot bars for each model
for i, (model_name, values) in enumerate(sensitivity_values.items()):
    ax.bar(x + i * width, values, width, label=model_name, color=colors[i])

# Formatting
ax.set_xlabel("TP (%)", fontsize=22)
ax.set_ylabel("Sensitivity (%)", fontsize=22)
ax.set_title("Comparative Analysis of Sensitivity", fontsize=22)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(x_labels, fontsize=20)
ax.set_ylim(0, 100)  # Y-axis from 0 to 100
ax.set_yticks(np.arange(0, 101, 20))  # Y-axis ticks at intervals of 20
ax.legend(loc='lower right', fontsize=17)

ax.tick_params(axis='y', labelsize=20)

# Show the plot
plt.show()



























