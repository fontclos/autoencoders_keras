# License
# Copyright 2018 Hamaad Musharaf Shah
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import math
import inspect

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, convolutional, pooling
from keras.models import Model

import tensorflow

from autoencoders_keras.loss_history import LossHistory


class Convolutional2DAutoencoder(BaseEstimator,
                                 TransformerMixin):
    def __init__(self,
                 input_shape=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 filters=None,
                 kernel_size=None,
                 strides=None,
                 pool_size=None,
                 denoising=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        loss_history = LossHistory()

        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10)

        reduce_learn_rate = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                              factor=0.1,
                                                              patience=20)

        self.callbacks_list = [loss_history, early_stop, reduce_learn_rate]

        # INPUTS
        self.input_data = Input(shape=self.input_shape)

        # ENCODER
        self.encoded = BatchNormalization()(self.input_data)
        for i in range(self.encoder_layers):
            self.encoded = BatchNormalization()(self.encoded)
            self.encoded = convolutional.Conv2D(
                filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.encoded)
            self.encoded = Dropout(rate=0.5)(self.encoded)
            self.encoded = pooling.MaxPooling2D(
                strides=self.pool_size, padding="same")(self.encoded)

        # DECODER
        self.decoded = BatchNormalization()(self.encoded)
        for i in range(self.decoder_layers):
            self.decoded = BatchNormalization()(self.decoded)
            self.decoded = convolutional.Conv2D(
                filters=self.filters, kernel_size=self.kernel_size, strides=self.strides, activation="elu", padding="same")(self.decoded)
            self.decoded = Dropout(rate=0.5)(self.decoded)
            self.decoded = convolutional.UpSampling2D(
                size=self.pool_size)(self.decoded)

        # ACTIVATION
        self.decoded = convolutional.Conv2D(
            filters=self.input_shape[2], kernel_size=self.kernel_size, strides=self.strides, activation="sigmoid", padding="same")(self.decoded)
        self.autoencoder = Model(self.input_data, self.decoded)
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                 loss="mean_squared_error")

    def fit(self,
            X,
            y=None):
        self.autoencoder.fit(X if self.denoising is None else X + self.denoising, X,
                             validation_split=0.3,
                             epochs=self.n_epoch,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=self.callbacks_list,
                             verbose=1)

        self.encoded_for_transformer = keras.layers.Flatten()(self.encoded)

        self.encoder = Model(self.input_data, self.encoded_for_transformer)

        return self

    def transform(self,
                  X):
        return self.encoder.predict(X)
