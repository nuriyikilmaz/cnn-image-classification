import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(images, labels, predictions, class_names, count=10):
    plt.figure(figsize=(10, 5))
    for i in range(count):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(labels[i])
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)
    plt.show()
