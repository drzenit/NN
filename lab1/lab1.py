import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential

# Загружаем данные из keras.mnist
mnist = tf.keras.datasets.mnist
# Разделяем данные на обучающие и тестовые
(feature_train, label_train), (feature_test, label_test) = mnist.load_data()

# Нормировка данных изображений
feature_train, feature_test = feature_train / 255, feature_test / 255
feature_train, feature_test = np.expand_dims(feature_train, axis=-1), np.expand_dims(feature_test, axis=-1)

# Инициализируем модель
model = Sequential()

# Создаем слои
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), padding="valid", activation=tf.nn.relu))
model.add(MaxPool2D((2, 2), (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3), padding="valid", activation=tf.nn.relu))
model.add(MaxPool2D((2, 2), (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding="valid", activation=tf.nn.relu))
model.add(Flatten())
model.add(Dense(10, activation=tf.nn.softmax))

# Компилируем модель
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Обучаем модель
model.fit(feature_train, label_train, epochs=5)

# Тестируем модель и выводим результаты
testLoss, testAccuracy = model.evaluate(feature_test, label_test)
print(f'Accuracy on TEST_data = {testAccuracy}')
