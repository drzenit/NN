import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras import models, layers, losses
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


# Загружаем данные из MNIST
(normal_x_train, _), (normal_x_test, _) = mnist.load_data()

# Нормировка данных изображений
normal_x_train = normal_x_train.reshape((len(normal_x_train), np.prod(normal_x_train.shape[1:])))
normal_x_test = normal_x_test.reshape((len(normal_x_test), np.prod(normal_x_test.shape[1:])))
normal_x_train = normal_x_train.astype('float32') / 255.0
normal_x_test = normal_x_test.astype('float32') / 255.0
normal_y_test = np.full((10000,), 1)

# Загружаем данные из FASHION_MNIST
(anomalous_x_train, _), (anomalous_x_test, _) = fashion_mnist.load_data()

# Нормировка данных изображений
anomalous_x_train = anomalous_x_train.reshape((len(anomalous_x_train), np.prod(anomalous_x_train.shape[1:])))
anomalous_x_test = anomalous_x_test.reshape((len(anomalous_x_test), np.prod(anomalous_x_test.shape[1:])))
anomalous_x_train = anomalous_x_train.astype('float32') / 255.0
anomalous_x_test = anomalous_x_test.astype('float32') / 255.0
anomalous_y_test = np.full((10000,), 0)

latent_dim = 128

# Класс для определения аномалий
class AnomalyDetector(models.Model):
    def __init__(self, latent_dim):
        super(AnomalyDetector, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.Input(784),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Создаем автокодировщик
autoencoder = AnomalyDetector(latent_dim)

# Компилируем модель
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# Обучаем модель на тренировачных данных
history = autoencoder.fit(
    normal_x_train,
    normal_x_train,
    epochs=10,
    shuffle=True,
    validation_data=(normal_x_test, normal_x_test)
)

# Строим графики потерь
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Кодируем и декодируем изображения
encoded_imgs = autoencoder.encoder(normal_x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# Вывод изображений
n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
    # Исходные изображения
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(normal_x_test[i].reshape(28, 28))
    plt.title(f'orig. {1}')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Раскодированные изображения
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.title(f'rec. {i}')
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# Определение аномалий в изображениях
reconstructions = autoencoder.predict(normal_x_train)
train_loss = tf.keras.losses.MSE(reconstructions, normal_x_train)

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)

# Функция предсказания
def predict(model, data, threshold):
    reconstructions = model(data)
    loss = tf.keras.losses.MSE(reconstructions, data)
    return tf.math.less(loss, threshold)

# Вывод результатов
def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))

# Перемешивание
def shuffle_in_unison(a, b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    return (a[indices], b[indices])


# Смешение изображений разных классов
x_test = np.concatenate((normal_x_test, anomalous_x_test))
y_test = np.concatenate((normal_y_test, anomalous_y_test))

# Случайным образом смешиваем данные
x_test, y_test = shuffle_in_unison(x_test, y_test)

# Получаем предсказания
pred = predict(autoencoder, x_test, threshold)

# Выводим статистику
print_stats(pred, y_test)
