import tensorflow as tf
import matplotlib.pyplot as plt

train_dataset = [65, 50, 55, 65, 55, 70, 65, 70]
train_labels = [85, 74, 76, 90, 85, 87, 94, 98]

def build_model():
  model = keras.Sequential(tf.keras.layers.Dense(1,input_dim = 1))
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)

  model.compile(loss='mse',
                optimizer=optimizer)
  return model

model = build_model()

model.fit(train_dataset,train_labels,epochs=1000)

print(model.predict(np.array([70])))