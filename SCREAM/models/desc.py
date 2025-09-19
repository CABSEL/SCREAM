import os
import math
from datetime import datetime
import random
import numpy as np
import pandas as pd
import json
import tensorflow as tf
import multiprocessing

from anndata import AnnData
from mudata import MuData
import scanpy as sc

from scipy.sparse import issparse

from .network import DescModel
from . import tools

os.environ['PYTHONHASHSEED'] = '0'
havedisplay = "DISPLAY" in os.environ


class DESC:
    def __init__(self, data, dims=None, use_rep=None, save_dir=None, logger=False):
        if isinstance(data, AnnData):
            self.data = data
            if use_rep is not None:
                assert use_rep in self.data.obsm_keys(), (f'Key {use_rep} not found in adata.obsm')

            self.use_rep = use_rep
        elif isinstance(data, MuData):
            self.data = data

            assert (use_rep is not None), ('If data is Mudata, use_rep needs to be provided of joint dimensions (mdata.obsm).')
            assert use_rep in self.data.obsm_keys(), (f'Key {use_rep} not found in mdata.obsm')
            self.use_rep = use_rep
        elif isinstance(data, pd.DataFrame):
            self.data = sc.AnnData(data, obs=data.index, var=data.columns)
            self.use_rep = None
        else:
            data_mat = data.toarray() if issparse(data) else data
            self.data = sc.AnnData(data_mat)
            self.use_rep = None

        if dims is None:
            if self.use_rep is None:
                self.dims = self.getdims(self.data.shape)
            else:
                self.dims = self.getdims(self.data.obsm[self.use_rep].shape)
        else:
            self.dims = dims

        if self.use_rep is None:
            assert self.dims[0] == self.data.shape[-1], (f'Mismatch in dimensions: expected dims[0] ({self.dims[0]}) to equal adata.shape[-1] ({self.data.shape[-1]}).')
        else:
            assert self.dims[0] == self.data.obsm[self.use_rep].shape[-1], (f'Mismatch in dimensions: expected dims[0] ({self.dims[0]}) to equal adata.shape[-1] ({self.data.obsm[self.use_rep].shape[-1]}).')

        self.save_dir = 'result_tmp' if save_dir is None else save_dir
        self.logger = logger

    @staticmethod
    def getdims(x=(10000, 200)):
        """
        This function will give the suggested nodes for each encoder layer
        return the dims for network
        """
        assert len(x) == 2
        n_sample = x[0]
        if n_sample > 20000:  # may be need complex network
            dims = [x[-1], 128, 32]
        elif n_sample > 10000:  # 10000
            dims = [x[-1], 64, 32]
        elif n_sample > 5000:  # 5000
            dims = [x[-1], 32, 16]  # 16
        elif n_sample > 2000:
            dims = [x[-1], 128]
        elif n_sample > 500:
            dims = [x[-1], 64]
        else:
            dims = [x[-1], 16]

        return dims

    @staticmethod
    def parseclusters(clustering_resolution):
        if isinstance(clustering_resolution, float) or isinstance(clustering_resolution, int):
            clustering_resolution = [float(clustering_resolution)]
        elif isinstance(clustering_resolution, str):
            clustering_resolution = list(map(float, clustering_resolution.split(",")))
        else:
            assert isinstance(clustering_resolution, list), ('clustering_resolution must be either a string with separator "," or a list like [1.0,2.0,3.0]')
            clustering_resolution = list(map(float, clustering_resolution))

        return clustering_resolution

    @staticmethod
    def random_init(seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def check_cores(ncores):
        total_cpu = multiprocessing.cpu_count()
        print(f'The number of available cores is {total_cpu}.')
        ncores = int(ncores) if total_cpu > int(ncores) else int(math.ceil(total_cpu/2))
        print(f'The number of cores used for training is {ncores}.')

        return ncores

    @staticmethod
    def configure_device(gpu_device=None):
        # Configure GPU usage
        if gpu_device is not None:
            try:
                # Set the specific GPU to use
                gpus = tf.config.list_physical_devices(device_type='GPU')
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if len(gpus) > 0:
                    tf.config.set_visible_devices(gpus[gpu_device], 'GPU')
                    tf.config.experimental.set_memory_growth(gpus[gpu_device], True)
                print(f'Training will be done on {gpus[gpu_device]}.')
            except RuntimeError as e:
                print(f"Error setting GPU configuration: {e}")
                os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # Fallback to CPU if GPU configuration fails
                print('Training will be done on CPU.')
        else:
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            print('Training will be done on CPU.')

    def train(self, clustering_resolutions, do_umap=True, umap_kwargs=None, do_tsne=True, tsne_kwargs=None, gpu_device=None, max_iter=1000, seed=201809, verbose=True, save_clustering_model=False, **desc_kwargs):
        clustering_resolutions = self.parseclusters(clustering_resolutions)

        temp_dict = {'dims': self.dims, 'max_iter': max_iter, 'seed': seed}

        default_desc_options = {'alpha': 1.0, 'tol': 0.005, 'init': 'glorot_uniform',
                                'method': 'leiden', 'n_neighbors': 10,
                                'pretrain_epochs': 300, 'epochs_fit': 5, 'batch_size': 256,
                                'random_seed': seed,
                                'activation': 'relu', 'actincenter': 'tanh',
                                'drop_rate_SAE': 0.2,
                                'use_earlyStop': True, 'pretrain_stacks': True, 'use_ae_weights': False,
                                'save_encoder_weights': False, 'save_encoder_step': 5,
                                'kernel_clustering': 't', 'suffix': ''}

        default_desc_options.update(desc_kwargs)
        print(desc_kwargs)
        temp_dict.update({'desc_opts': default_desc_options})

        if self.logger is True:
            default_desc_options.update({'logger': True})

        if do_umap is True:
            default_neigh_kwargs = {'n_neighbors': default_desc_options['n_neighbors']}
            default_umap_kwargs = {}
            if umap_kwargs is not None:
                default_umap_kwargs.update(umap_kwargs)

                print(umap_kwargs)
            temp_dict.update({'umap_opts': default_umap_kwargs})

        if do_tsne is True:
            default_tsne_kwargs = {'n_jobs': 10, 'learning_rate': 150, 'perplexity': 30}
            if tsne_kwargs is not None:
                default_tsne_kwargs.update(tsne_kwargs)

                print(tsne_kwargs)

            default_tsne_kwargs['n_jobs'] = self.check_cores(default_tsne_kwargs['n_jobs'])
            temp_dict.update({'tsne_opts': default_tsne_kwargs})

        t_start = datetime.now()
        for i, resolution in enumerate(clustering_resolutions):
            print(f'Start to process resolution: {str(resolution)}')
            self.random_init(seed)

            if i > 0:
                default_desc_options.update({'use_ae_weights': True})

            ae_save_loc = os.path.join(self.save_dir, "ae.weights.h5")
            if not default_desc_options['use_ae_weights'] and os.path.isfile(ae_save_loc):
                os.remove(ae_save_loc)

            model = self.train_resolution(resolution=resolution, gpu_device=gpu_device, max_iter=max_iter, **default_desc_options)

            if verbose is True:
                print("The summary of desc model is:\n")
                model.model.summary()

            if save_clustering_model is True:
                model.model.save(os.path.join(self.save_dir, 'desc_'+str(resolution)+'_clustermodel.keras'))

            if do_umap is True:
                self.umap(self.data, resolution=resolution, neigh_kwargs=default_neigh_kwargs, **default_umap_kwargs)

            if do_tsne is True:
                self.tsne(self.data, resolution=resolution, **default_tsne_kwargs)

        with open(os.path.join(self.save_dir, 'desc_hparams.json'), 'w') as f:
            json.dump(temp_dict, f)

        print(f'Total time to run DESC (HH:MM:SS): {str(datetime.now() - t_start)}')

        return model

    def train_resolution(self, resolution, gpu_device=None, max_iter=1000, **default_desc_options):
        self.configure_device(gpu_device)
        t_epoch = datetime.now()
        print(f'Runtime for resolution {resolution} (HH:MM:SS): {str(datetime.now() - t_epoch)}')

        if self.use_rep is None:
            x = self.data.X
        else:
            x = self.data.obsm[self.use_rep]

        desc = DescModel(dims=self.dims, x=x, clustering_resolution=resolution, save_dir=self.save_dir, **default_desc_options)
        desc.compile(optimizer=tf.keras.optimizers.SGD(0.01, 0.9), loss='kld')
        Embedded_z, q_pred = desc.fit(maxiter=max_iter)
        print("DESC has been trained successfully!!!!!!")

        y_pred = pd.Series(np.argmax(q_pred, axis=1), index=self.data.obs.index, dtype='category')
        y_pred = y_pred.cat.rename_categories(range(0, len(y_pred.cat.categories)))

        self.data.obs['desc_'+str(resolution)] = y_pred
        self.data.obsm['X_Embedded_z'+str(resolution)] = Embedded_z
        self.data.uns['prob_matrix'+str(resolution)] = q_pred

        return desc

    def predict_ls(self, adata, model, use_rep=None):
        if isinstance(model, str):
            model = tools.load_model(model)
        else:
            assert isinstance(model, tf.keras.Model), ('Encoder provided isnt a keras model')

        if use_rep is not None:
            assert use_rep in adata.obsm_keys(), (f'Key {use_rep} not found in adata.obsm')
            x = adata.obsm[use_rep].copy()
        else:
            x = adata.X.copy()

        assert x.shape[1] == model.input_shape[1], (f'Encoder input expects {model.input_shape[1]} features while {x.shape[1]} was provided')
        ls_mat = model.predict(x)

        return ls_mat

    def predict_embedding(self, adata, encoder, use_rep=None, key_added='X_desc_ls'):
        x = self.predict_ls(adata, model=encoder, use_rep=use_rep)
        adata.obsm[key_added] = x.copy()

        return adata

    def predict_reconstructed(self, adata, autoencoder, use_rep=None, key_added='desc_reconst'):
        x = self.predict_ls(adata, model=autoencoder, use_rep=use_rep)
        adata.layers[key_added] = x.copy()

        return adata

    def predict_clusters(self, adata, cluster_model, use_rep=None, key_added='cluster'):
        q = self.predict_ls(adata, model=cluster_model, use_rep=use_rep)
        adata.obs[key_added] = q.argmax(axis=1)

        return adata

    @staticmethod
    def umap(mdata, resolution, neigh_kwargs=None, **umap_kwargs):
        sc.pp.neighbors(mdata, use_rep='X_Embedded_z'+str(resolution), key_added=None, **neigh_kwargs)  # neigh_kwds shouldn't have key_added, copy
        sc.tl.umap(mdata, neighbors_key='neighbors', **umap_kwargs)  # umap_kwds shouldn't have neighbors_key, copy
        mdata.obsm['X_umap'+str(resolution)] = mdata.obsm['X_umap'].copy()
        print(f"UMAP finished and added X_umap{resolution} into adata.obsm\n")
        del mdata.uns['neighbors']
        del mdata.obsp['distances'], mdata.obsp['connectivities']

    @staticmethod
    def tsne(mdata, resolution, **tsne_kwargs):
        sc.tl.tsne(mdata, use_rep="X_Embedded_z"+str(resolution), **tsne_kwargs)
        mdata.obsm["X_tsne"+str(resolution)] = mdata.obsm["X_tsne"].copy()
        print(f"TSNE finished and added X_tsne{resolution} into adata.obsm\n")