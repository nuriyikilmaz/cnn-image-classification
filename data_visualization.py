import matplotlib.pyplot as plt
import numpy as np

def plot_sample_images(images, labels, class_names, count=10):
    plt.figure(figsize=(10, 5))
    for i in range(count):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[np.argmax(labels[i])])
    plt.show()
