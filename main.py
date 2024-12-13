from data_processing import load_and_process_data
from data_visualization import plot_sample_images
from model_definition import create_model
from model_training import train_and_plot_model
from prediction_visualization import plot_predictions

# 1. Veri setini yükleme ve işleme
x_train, y_train, x_test, y_test, class_names = load_and_process_data()

# 2. Veri görselleştirme
plot_sample_images(x_train, y_train, class_names)

# 3. Model oluşturma
model = create_model()
model.summary()

# 4. Model eğitimi ve doğrulama
train_and_plot_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=32)

# 5. Tahminleri görselleştirme
predictions = model.predict(x_test[:10])
plot_predictions(x_test[:10], y_test[:10], predictions, class_names)
