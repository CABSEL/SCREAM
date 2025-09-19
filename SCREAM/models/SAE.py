import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from functools import partial
from .tools import CustomReduceLROnPlateau


class SAE_base:
    """
    Stacked autoencoders
    """

    def __init__(self, dims, act='relu', drop_rate=0.2, batch_size=32, random_seed=201809, actincenter="tanh", init="glorot_uniform", use_earlyStop=True, pretrain_stacks=True, suffix=''):
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.drop_rate = drop_rate
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.activation = act
        self.actincenter = actincenter
        self.init = init
        self.use_earlyStop = use_earlyStop
        self.stack_pretrain = pretrain_stacks
        self.suffix = suffix

        self.random_init(self.random_seed)  # Set random seed

        # Create autoencoders
        if self.stack_pretrain is True:
            self.stacks = [self.make_stack(i, random_seed=self.random_seed + 2 * i) for i in np.arange(self.n_stacks)]

        self.autoencoders, self.encoder = self.make_autoencoders()

    @staticmethod
    def random_init(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def choose_init(init='glorot_uniform', seed=1):
        seed = int(seed)
        valid_inits = {'glorot_uniform': tf.keras.initializers.GlorotUniform,
                       'glorot_normal': tf.keras.initializers.GlorotNormal,
                       'he_normal': tf.keras.initializers.HeNormal,
                       'he_uniform': tf.keras.initializers.HeUniform,
                       'lecun_normal': tf.keras.initializers.LecunNormal,
                       'lecun_uniform': tf.keras.initializers.LecunUniform,
                       'RandomNormal': tf.keras.initializers.RandomNormal,
                       'RandomUniform': tf.keras.initializers.RandomUniform,
                       'TruncatedNormal': tf.keras.initializers.TruncatedNormal}

        if init not in valid_inits:
            raise ValueError(f'Invalid `init` argument: expected one of {list(valid_inits.keys())} but got {init}')

        return valid_inits[init](seed=seed)

    def make_autoencoders(self):
        """
        Fully connected autoencoders model, symmetric
        """
        # input
        x = Input(shape=(self.dims[0],), name=f'input{self.suffix}')

        h = x
        # internal layers in encoder
        for i in np.arange(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+i), activation=self.activation, name=f'encoder_{i}{self.suffix}')(h)

        h = Dense(self.dims[-1], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+self.n_stacks), activation=self.actincenter, name=f'encoder_{self.n_stacks - 1}{self.suffix}')(h)  # last encoder layer. features are extracted from here

        y = h
        # internal layers in decoder
        for i in np.arange(self.n_stacks-1, 0, -1):
            y = Dense(self.dims[i], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+self.n_stacks+i), activation=self.activation, name=f'decoder_{i}{self.suffix}')(y)

        y = Dense(self.dims[0], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+2*self.n_stacks), activation=self.actincenter, name=f'decoder_0{self.suffix}')(y)  # output

        # Build models
        autoencoder = Model(inputs=x, outputs=y, name=f"AE{self.suffix}")
        encoder = Model(inputs=x, outputs=h, name=f"encoder{self.suffix}")

        return autoencoder, encoder

    def make_stack(self, ith, random_seed=1234):
        """
        Create the i-th denoising autoencoder for layer-wise pretraining
        """
        random_seed = int(random_seed) # Do not know why tf doesnt accept np.int64
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith + 1]
        output_act = self.activation if ith != 0 else self.actincenter
        hidden_act = self.activation if ith != self.n_stacks - 1 else self.actincenter

        # Define model layers
        input_layer = Input(shape=(in_out_dim,), name='input')
        dropout_layer = Dropout(self.drop_rate, seed=random_seed)(input_layer)
        encoder_layer = Dense(units=hidden_dim, activation=hidden_act,
                              kernel_initializer=self.choose_init(init=self.init, seed=random_seed),
                              name='encoder')(dropout_layer)
        dropout_layer_1 = Dropout(self.drop_rate, seed=random_seed+1)(encoder_layer)
        decoder_layer = Dense(units=in_out_dim, activation=output_act,
                              kernel_initializer=self.choose_init(init=self.init, seed=random_seed+1),
                              name='decoder')(dropout_layer_1)

        # Create model
        model = Model(inputs=input_layer, outputs=decoder_layer, name=f'stack_{ith}')

        return model

    def pretrain_stacks(self, x, epochs=200, decaying_step=3):
        """
        Layer-wise pretraining
        """
        assert (isinstance(decaying_step, int) and (decaying_step > 0)), ('Decaying step needs to be a positive integer.')
        features = x.astype('float32')
        for i in np.arange(self.n_stacks):
            print(f'Pretraining layer {i + 1}...')
            for step in np.arange(decaying_step):
                lr = np.float_power(10, (-1 - step))
                print(f'Learning rate = {lr}')
                self.stacks[i].compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
                callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')] if self.use_earlyStop else None
                self.stacks[i].fit(features, features, callbacks=callbacks, batch_size=self.batch_size, epochs=np.ceil(epochs / decaying_step).astype(int))

            print(f'Layer {i + 1} has been pretrained.')

            # Update features to the inputs of the next layer
            encoder = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder').output)
            features = encoder.predict(features)

    def fit_unstacked(self, x, epochs=300, callbacks=None):  # no stack directly train
        """
        Train the non-stacked autoencoder end-to-end
        """
        print('Fine-tuning autoencoder end-to-end')
        for step in np.arange(np.ceil(epochs/50).astype(int)):
            lr = np.float_power(10, (-1 - step))
            print(f'Learning rate = {lr}')
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')] if self.use_earlyStop else None
            self.autoencoders.fit(x=x, y=x, callbacks=callbacks, batch_size=self.batch_size, epochs=50)

    def fit(self, x, epochs=300, decaying_step=3, callbacks=None):
        """
        Train the stacked autoencoder optioally with pretraining and fine-tuning
        """
        if self.stack_pretrain is True:
            self.pretrain_stacks(x, epochs=int(epochs / 2), decaying_step=decaying_step)
            print('Copying layer-wise pretrained weights to deep autoencoders')

            for i in np.arange(self.n_stacks):
                self.autoencoders.get_layer(f'encoder_{i}{self.suffix}').set_weights(self.stacks[i].get_layer('encoder').get_weights())
                self.autoencoders.get_layer(f'decoder_{i}{self.suffix}').set_weights(self.stacks[i].get_layer('decoder').get_weights())

        self.fit_unstacked(x, epochs=epochs, callbacks=callbacks)

    def predict_ls(self, x):
        """
        Extract features from the encoder
        """
        return self.encoder.predict(x)

    def predict_reconstructed(self, x):
        """
        Make predictions using the autoencoder
        """
        return self.autoencoders.predict(x)

    def __repr__(self):
        model_text = f'SAE Keras model with {self.n_stacks} layers.'

        return model_text



def scheduler(epoch, lr, factor, decay_step, min_lr):
    return np.max([np.round(lr * factor, 6), min_lr]) if epoch > 0 and epoch % decay_step == 0 else lr


class SAE(SAE_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pretrain_stacks(self, x, epochs=200, decaying_step=3):
        """
        Layer-wise pretraining
        """
        assert (isinstance(decaying_step, int) and (decaying_step > 0)), ('Decaying step needs to be a positive integer.')
        features = x.astype('float32')
        init_lr = np.float_power(10, -1)
        min_lr = np.float_power(10, -decaying_step)
        for i in np.arange(self.n_stacks):
            print(f'Pretraining layer {i + 1}...')
            self.stacks[i].compile(optimizer=SGD(init_lr, momentum=0.9), loss='mse')
            callbacks = []
            if self.use_earlyStop:
                callbacks.append(CustomReduceLROnPlateau(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto', max_lr_epochs=np.ceil(epochs / decaying_step).astype(int), min_lr=min_lr))
            else:
                callbacks.append(LearningRateScheduler(partial(scheduler, factor=0.1, decay_step=np.ceil(epochs / decaying_step).astype(int), min_lr=min_lr)))
            self.stacks[i].fit(features, features, callbacks=callbacks, batch_size=self.batch_size, epochs=epochs)
            print(f'Layer {i + 1} has been pretrained.')

            # Update features to the inputs of the next layer
            encoder = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder').output)
            features = encoder.predict(features)

    def fit_unstacked(self, x, epochs=300, callbacks=None):  # no stack directly train
        """
        Train the non-stacked autoencoder end-to-end
        """
        print('Fine-tuning autoencoder end-to-end')
        decay_epochs = 50
        init_lr = np.float_power(10, -1)
        min_lr = np.float_power(10, -np.ceil(epochs/decay_epochs).astype(int))
        self.autoencoders.compile(optimizer=SGD(init_lr, momentum=0.9), loss='mse')
        if callbacks is None:
            callbacks = []
        else:
            assert isinstance(callbacks, list), ('Callbacks must be provided as a list.')

        if self.use_earlyStop:
            callbacks.append(CustomReduceLROnPlateau(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto', max_lr_epochs=decay_epochs, min_lr=min_lr))
        else:
            callbacks.append(LearningRateScheduler(partial(scheduler, factor=0.1, decay_step=decay_epochs, min_lr=min_lr)))

        self.autoencoders.fit(x=x, y=x, callbacks=callbacks, batch_size=self.batch_size, epochs=epochs)