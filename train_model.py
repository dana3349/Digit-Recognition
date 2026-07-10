import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# تحميل البيانات
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# التطبيع
x_train = x_train / 255.0
x_test = x_test / 255.0

# إنشاء النموذج
model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# التدريب
model.fit(
    x_train,
    y_train,
    epochs=10,
    validation_data=(x_test,y_test)
)

# التقييم
loss, acc = model.evaluate(x_test,y_test)

print("Accuracy:",acc)

# الحفظ
model.save("model.h5")