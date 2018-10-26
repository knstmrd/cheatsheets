import keras
import numpy as np


class DummyDataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, feature_dim, train=True):
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        # self.x_data = np.random.random((100, 100))
        self.x_data = np.random.random((100, 100))
        self.train = train
        self.y_data = np.zeros((self.x_data.shape[0], 10))
        for i in range(10):
            self.y_data[:, i] = self.x_data.sum(axis=1) * i
        # self.y_data = self.y_data > 1
        # self.y_data = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, idx):
        return self.x_data[idx*self.batch_size:(idx + 1)*self.batch_size, :], self.y_data[idx*self.batch_size:(idx + 1)*self.batch_size, :]


train_gen = DummyDataGenerator(32, 100)
test_gen = DummyDataGenerator(32, 100, False)

print(train_gen[1][1].shape)


model = keras.models.Sequential()
model.add(keras.layers.Dense(units=100, activation='relu', input_dim=100))
model.add(keras.layers.Dense(units=10))

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.99, nesterov=True),
              metrics=[keras.metrics.mean_squared_error])


# checkp = keras.callbacks.ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
#                                        save_best_only=False, save_weights_only=False, mode='auto', period=1)
es = keras.callbacks.EarlyStopping(monitor='train_loss', min_delta=0, patience=2, verbose=0, mode='auto')


model.fit(train_gen.x_data, train_gen.y_data, epochs=10, batch_size=32, validation_split=0.1, callbacks=[es])
# model.fit_generator(train_gen, epochs=2, callbacks=[es], workers=2, use_multiprocessing=True)

# score = model.evaluate(x_test, y_test, batch_size=128)
score = model.evaluate_generator(test_gen)
print(score)
