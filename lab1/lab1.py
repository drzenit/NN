from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from  tensorflow.keras.datasets import mnist


def firstTask():
    # Задаем размерность
    imgRows, imgCols = 28, 28

    # Загружаем разделенные данные из kears.mnist
    (feature_train, label_train), (feature_test, label_test) = mnist.load_data()
    print("Размер тренировачных данных - ", len(feature_train))
    print("Размер тестовых данных - ", len(feature_train))

    # Нормировка данных изображений
    feature_train = feature_train / 255
    feature_test = feature_test / 255

    # Инициализируем нейронную сеть
    model = Sequential()

    # Добавляем и настраиваем слои
    model.add(Flatten(input_shape=(imgRows, imgCols)))  # Создаем входной слой равный размерности изображения (28 * 28 = 784)
    model.add(Dense(128, activation="relu"))  # Добавляем скрытый слой
    model.add(Dense(10, activation="softmax"))  # Добавляем выходной слой на 10 нейронов = количеству типов цифр [0-9]

    # Компилируем модель
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Обучаем модель
    model.fit(feature_train, label_train, epochs=5)

    # Тестирование модели методами keras
    testLoss, testAccuracy = model.evaluate(feature_test, label_test)
    print("Точность  = ", testAccuracy)


firstTask()
