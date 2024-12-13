import matplotlib.pyplot as plt

def train_and_plot_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=32):
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=batch_size
    )

    # Eğitim sürecini görselleştirme
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend(loc='lower right')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.show()

    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend(loc='upper right')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.show()
