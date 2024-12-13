from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def load_and_process_data():
    # CIFAR-10 veri setini yükleme
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalizasyon
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Etiketleri kategorik hale getirme (one-hot encoding)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # CIFAR-10 sınıf isimleri
    class_names = ['Uçak', 'Araba', 'Kuş', 'Kedi', 'Geyik', 'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']

    return x_train, y_train, x_test, y_test, class_names
