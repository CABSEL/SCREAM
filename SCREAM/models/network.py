"""
Keras implement Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis
"""
# from __future__ import division
import os
from datetime import datetime
import numpy as np
import random
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.models import Model

import tensorflow as tf
import scanpy as sc
import pandas as pd

from .SAE import SAE  # this is for installing package

havedisplay = "DISPLAY" in os.environ

os.environ['PYTHONHASHSEED'] = '0'

random.seed(201809)
np.random.seed(201809)
tf.random.set_seed(201809)


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')  # the first parameter is shape and not name
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClusteringLayerGaussian(ClusteringLayer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super().__init__(n_clusters, weights, alpha, **kwargs)

    def call(self, inputs, **kwargs):
        sigma = 1.0
        q = K.sum(K.exp(-K.square(K.expand_dims(inputs, axis=1)-self.clusters)/(2.0*sigma*sigma)), axis=2)
        q = K.transpose(K.transpose(q)/K.sum(q, axis=1))
        return q


class DescModel:
    def __init__(self,
                 dims,
                 x,  # input matrix, row sample, col predictors
                 alpha=1.0,
                 tol=0.005,
                 init='glorot_uniform',  # initialization method
                 method='leiden',
                 clustering_resolution=1.0,  # resolution for clustering
                 n_neighbors=10,    # the
                 pretrain_epochs=300,  # epoch for autoencoder
                 epochs_fit=4,  # epochs for each update,int or float
                 batch_size=256,  # batch_size for autoencoder
                 random_seed=201809,
                 activation='relu',
                 actincenter="tanh",  # activation for the last layer in encoder, and first layer in the decoder
                 drop_rate_SAE=0.2,
                 use_earlyStop=True,
                 pretrain_stacks=True,
                 use_ae_weights=False,
                 save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp",
                 kernel_clustering="t",
                 suffix='',
                 logger=False):
        '''
        save result to save_dir, the default is "result_tmp". if recursive path, the root dir must be exists, or there will be something wrong.
        For example : "/result_singlecell/dataset1" will return wrong if "result_singlecell" not exist
        '''

        if not os.path.exists(save_dir):
            print(f"Create the directory: {save_dir}  to save result")
            os.mkdir(save_dir)

        self.dims = dims
        self.x = x  # feature n*p, n:number of cells, p: number of genes
        self.alpha = alpha
        self.tol = tol
        self.init = init
        self.method = method
        self.resolution = clustering_resolution
        self.n_neighbors = n_neighbors
        self.pretrain_epochs = pretrain_epochs
        self.epochs_fit = epochs_fit
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.activation = activation
        self.actincenter = actincenter
        self.drop_rate_SAE = drop_rate_SAE
        self.use_earlyStop = use_earlyStop
        self.pretrain_stacks = pretrain_stacks
        self.use_ae_weights = use_ae_weights
        self.save_encoder_weights = save_encoder_weights
        self.save_encoder_step = save_encoder_step
        self.save_dir = save_dir
        self.kernel_clustering = kernel_clustering
        self.suffix = suffix
        self.logger = logger
        self.input_dim = dims[0]  # for clustering layer
        self.n_stacks = len(self.dims) - 1

        # set random seed
        self.random_init(random_seed)

        # pretrain autoencoder
        self.pretrain()

    @staticmethod
    def random_init(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def train_sae(self):
        sae = SAE(dims=self.dims,
                  act=self.activation,
                  drop_rate=self.drop_rate_SAE,
                  batch_size=self.batch_size,
                  random_seed=self.random_seed,
                  actincenter=self.actincenter,
                  init=self.init,
                  use_earlyStop=self.use_earlyStop,
                  pretrain_stacks=self.pretrain_stacks)

        # begin pretraining
        t0 = datetime.now()

        ae_weights_file = os.path.join(self.save_dir, 'ae.weights.h5')
        if self.use_ae_weights and os.path.isfile(ae_weights_file):
            sae.autoencoders.load_weights(ae_weights_file)
        else:
            print(f"Rerunning autoencoder{' (file ae.weights.h5 doesnt exist)' if self.use_ae_weights else ''}")
            if self.logger is True:
                print(f"Logged at {os.path.join(self.save_dir, 'pretrain')}")
                pretrain_callbacks = [tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.save_dir, 'pretrain'), update_freq='epoch')]

            sae.fit(self.x, epochs=self.pretrain_epochs, callbacks=pretrain_callbacks)

        self.autoencoder = sae.autoencoders
        self.encoder = sae.encoder  # Is this correct? If weights loaded to autoencoder, will encoder take from it (For case where ae.weights.h5 is loaded)?

        print(f'Pretraining time (HH:MM:SS): {datetime.now() - t0}')

        # save ae results into disk
        if not os.path.isfile(ae_weights_file):
            self.autoencoder.save_weights(ae_weights_file)
            self.encoder.save_weights(os.path.join(self.save_dir, 'encoder.weights.h5'))
            print(f'Pretrained weights are saved to {ae_weights_file}')

        # save autoencoder model
        self.autoencoder.save(os.path.join(self.save_dir, "autoencoder_model.keras"))

    def pretrain(self):  # , use_clus_centroids=None
        self.train_sae()

        # initialize cluster centers using clustering if n_clusters is not exist
        features = self.predict_ls(self.x)
        features = np.asarray(features)
        feat_dims = features.shape[0]

        features_df = pd.DataFrame(features, index=np.arange(0, feat_dims))
        if feat_dims > 200000:
            features_df = features_df.sample(n=200000, replace=False, axis='index', ignore_index=True, random_state=feat_dims)

        print(f'...number of clusters is unknown, Initialize cluster centroid using {self.method} method')

        # can be replaced by other clustering methods
        if self.method == 'leiden':
            adata0 = sc.AnnData(features_df)
            sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors, use_rep="X")
            sc.tl.leiden(adata0, resolution=self.resolution)
            Y_pred_init = adata0.obs[self.method].astype(int)
        elif self.method == 'louvain':
            adata0 = sc.AnnData(features_df)
            sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors, use_rep="X")
            sc.tl.louvain(adata0, resolution=self.resolution)
            Y_pred_init = adata0.obs[self.method].astype(int)
        else:
            # saved for other initizlization methods in futher
            # print("...number of clusters have been specified, Initializing Cluster centroid  using K-Means")
            """
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            Y_pred_init = kmeans.fit_predict(features_df)
            """
        assert Y_pred_init.dtype == int
        if Y_pred_init.unique().shape[0] <= 1:
            raise ValueError(f'Error: Only one cluster detected. The resolution ({self.resolution}) is too small. Please choose a larger resolution.')

        self.init_pred = Y_pred_init.copy()
        features_df['Group'] = Y_pred_init.values

        cluster_centers = np.asarray(features_df.groupby('Group').mean())
        self.n_clusters = cluster_centers.shape[0]
        self.init_centroid = [cluster_centers]

        # create desc clustering layer
        if self.kernel_clustering == "gaussian":
            clustering_layer = ClusteringLayerGaussian(self.n_clusters, weights=self.init_centroid, name=f'clustering{self.suffix}')(self.encoder.output)
        else:
            clustering_layer = ClusteringLayer(self.n_clusters, weights=self.init_centroid, name=f'clustering{self.suffix}')(self.encoder.output)

        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def predict_ls(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_on_all(self, maxiter=1e3, epochs_fit=5, save_encoder_step=5):  # unsupervised
        # step1 initial weights by leiden,louvain or Kmeans
        if self.logger is True:
            summary_writer = tf.summary.create_file_writer(os.path.join(self.save_dir, 'finetune', self.method+'_'+str(self.resolution)))
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite % save_encoder_step == 0:  # save ae_weights for every 5 iterations
                encoder_savedir = os.path.join(self.save_dir, 'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.weights.h5')
                self.encoder.save_weights(encoder_savedir)
                print(f'Fine tuning encoder weights are saved to {encoder_savedir}')

            q = self.model.predict(self.x, verbose=0)
            p = self.target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            with summary_writer.as_default():
                tf.summary.scalar('delta_label', delta_label, step=ite)

            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < self.tol:
                print(f'delta_label {delta_label} < tol {self.tol}')
                print('Reached tolerance threshold. Stop training.')
                break

            print(f"The value of delta_label of current {ite+1}th iteration is: {delta_label} >= tol {self.tol}")
            # train on whole dataset on prespecified batch_size
            callbacks = [EarlyStopping(monitor='loss', min_delta=1e-4, patience=5, verbose=1, mode='auto')] if self.use_earlyStop else None
            self.model.fit(x=self.x, y=p, epochs=epochs_fit, batch_size=self.batch_size, callbacks=callbacks, shuffle=True, verbose=True)

        self.encoder.save(os.path.join(self.save_dir, "encoder_model.keras"))  # Save final encoder model

        y0 = pd.Series(y_pred, dtype='category')
        y0 = y0.cat.rename_categories(range(0, len(y0.cat.categories)))
        print(f"The final prediction cluster is:\n{y0.value_counts(sort=False)}")

        Embedded_z = self.predict_ls(self.x)

        return Embedded_z, q

    def fit(self, maxiter=1e4):
        embedded_z, q = self.fit_on_all(maxiter=maxiter, epochs_fit=self.epochs_fit, save_encoder_step=self.save_encoder_step)

        return embedded_z, q

    def __repr__(self):
        model_text = 'DESC Keras model.'

        return model_text