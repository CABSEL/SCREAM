import os
import numpy as np
import random
import tensorflow as tf
import math
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model  # , Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping


class SAE:
    """
    Stacked autoencoders
    """

    def __init__(self, dims, act="relu", drop_rate=0.2, batch_size=32, random_seed=201809, actincenter="tanh", init="glorot_uniform", use_earlyStop=True, save_dir='./result_tmp'):
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.activation = act
        self.actincenter = actincenter
        self.drop_rate = drop_rate
        self.init = init
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.use_earlyStop = use_earlyStop

        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        # Create output folder if not exists
        os.makedirs(save_dir, exist_ok=True)

        # Create autoencoders
        self.stacks = [self.make_stack(i, random_seed=self.random_seed + 2 * i) for i in range(self.n_stacks)]
        print(self.stacks)

        self.autoencoders, self.encoder = self.make_autoencoders()

#        plot_model(self.autoencoders, show_shapes=True, to_file=os.path.join(save_dir, 'autoencoders.png'))

    def choose_init(self, init="glorot_uniform", seed=1):
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
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+i), activation=self.activation, name='encoder_%d' % i)(h)

        # last encoder layer
        h = Dense(self.dims[-1], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+self.n_stacks), name='encoder_%d' % (self.n_stacks - 1), activation=self.actincenter)(h)  # features are extracted from here

        y = h
        # internal layers in decoder
        for i in range(self.n_stacks-1, 0, -1):
            y = Dense(self.dims[i], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+self.n_stacks+i), activation=self.activation, name='decoder_%d' % i)(y)

        # output
        y = Dense(self.dims[0], kernel_initializer=self.choose_init(init=self.init, seed=self.random_seed+2*self.n_stacks), name='decoder_0', activation=self.actincenter)(y)

        return Model(inputs=x, outputs=y, name="AE"), Model(inputs=x, outputs=h, name="encoder")

    def make_stack(self, ith, random_seed=1234):
        """
        Create the i-th denoising autoencoder for layer-wise pretraining
        """
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
        model = Model(inputs=input_layer, outputs=decoder_layer)

        return model

    def pretrain_stacks(self, x, epochs=200, decaying_step=3):
        """
        Layer-wise pretraining
        """
        features = x.astype('float32')
        for i in range(self.n_stacks):
            print(f'Pretraining layer {i + 1}...')
            for j in range(int(decaying_step)):
                lr = pow(10, -1 - j)
                print(f'Learning rate = {lr}')
                self.stacks[i].compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
                callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')] if self.use_earlyStop else []
                self.stacks[i].fit(features, features, callbacks=callbacks, batch_size=self.batch_size, epochs=math.ceil(epochs / decaying_step))
            print(f'Layer {i + 1} has been pretrained.')

            # Update features to the inputs of the next layer
            encoder = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder').output)
            features = encoder.predict(features)

    def pretrain_autoencoders(self, x, epochs=300):
        """
        Fine-tune autoencoders end-to-end after layer-wise pretraining
        """
        print('Copying layer-wise pretrained weights to deep autoencoders')

        for i in range(self.n_stacks):
            self.autoencoders.get_layer(f'encoder_{i}').set_weights(self.stacks[i].get_layer('encoder').get_weights())
            self.autoencoders.get_layer(f'decoder_{i}').set_weights(self.stacks[i].get_layer('decoder').get_weights())

        print('Fine-tuning autoencoder end-to-end')
        for j in range(math.ceil(epochs / 50)):
            lr = pow(10, -1-j)
            print(f'Learning rate = {lr}')
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            if self.use_earlyStop:
                callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')]
                self.autoencoders.fit(x=x, y=x, callbacks=callbacks, batch_size=self.batch_size, epochs=50)
            else:
                self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def fit(self, x, epochs=300, decaying_step=3):
        """
        Train the stacked autoencoder with pretraining and fine-tuning
        """
        self.pretrain_stacks(x, epochs=int(epochs / 2), decaying_step=decaying_step)
        self.pretrain_autoencoders(x, epochs=epochs)

    def fit2(self, x, epochs=300):  # no stack directly train
        """
        Train the non-stacked autoencoder directly
        """
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -1-j)
            print(f'Learning rate = {lr}')
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            if self.use_earlyStop:
                callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')]
                self.autoencoders.fit(x=x, y=x, callbacks=callbacks, batch_size=self.batch_size, epochs=epochs)
            else:
                self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders
        """
        return self.encoder.predict(x)

    def make_pred(self, x):
        """
        Make predictions using the autoencoder
        """
        return self.autoencoders.predict(x)


if __name__ == "__main__":
    def load_mnist(sample_size=10000):
        """Load and preprocess the MNIST dataset."""
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test)).reshape((-1, 28 * 28))
        y = np.concatenate((y_train, y_test))
        print('MNIST samples', x.shape)

        id0 = np.random.choice(x.shape[0], sample_size, replace=False)
        return x[id0], y[id0]

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    x, y = load_mnist(10000)
    print('Shape of X', x.shape)
    db = 'mnist'
    n_clusters = 10

    # Define and train SAE model
    sae = SAE(dims=[x.shape[-1], 64, 32])
    sae.fit(x=x, epochs=400)
    sae.autoencoders.save_weights(f'{db}.weights.h5')

    # Extract features
    print('Finished training, extracting features using the trained SAE model')
    features = sae.extract_feature(x)
    print('Performing k-means clustering on the extracted features')

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters, n_init=20)
    y_pred = km.fit_predict(features)

    from sklearn.metrics import normalized_mutual_info_score as nmi
    print('K-means clustering result on extracted features: NMI =', nmi(y, y_pred))