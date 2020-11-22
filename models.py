import tensorflow as tf

def MLP(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(200),
        ])

    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.summary()
    return model