import os
import pandas as pd
import numpy as np
import json
import tensorflow as tf
# import desc_leiden_SV as desc
# import mudata as md
import muon as mu

from .desc import DESC
from .SAE import SAE
from . import tools

mu.set_options(pull_on_update=False)


class SCREAM:
    def __init__(self, modality_data, save_dir=None, logger=False):
        self.mdata_train = mu.MuData(modality_data)
        self.mdata_train.pull_obs()
        mu.pp.intersect_obs(self.mdata_train)

        save_dir = 'result_tmp' if save_dir is None else save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        self.save_dir = save_dir
        print(f'Trained models will be saved at: {self.save_dir}')

        self.logger = logger

        print(self.mdata_train)

    def __repr__(self):
        header_text = 'SCREAM model with mdata:\n'
        mdata_text = self.mdata_train.__repr__()
        return header_text+mdata_text

    def train_test_split(self, X, split=None):
        if split is None:  # To be expanded for other case
            X_train = X.copy()
            # X_test = X_all_data.copy()

            return X_train, None, None

    def pretrain_modality(self, modality, encoding_layer_dims, train_test_split=None, save_encoder=True, epochs=300, decaying_step=3, **sae_kwargs):
        assert modality in self.mdata_train.mod_names, (f'Modality {modality} not found in mudata. Must be one of {self.mdata_train.mod_names}.')
        assert (isinstance(encoding_layer_dims, list) and len(encoding_layer_dims) > 0), ('Encoding layer dimensions needs to be non-empty list.')
        assert all(isinstance(e, int) for i, e in enumerate(encoding_layer_dims)), (('Encoding layer dimensions needs to be integers.'))

        X_all_data = self.mdata_train.mod[modality].X.copy()
        X_train, X_val, X_test = self.train_test_split(X_all_data)

        default_sae_kwargs = {'act': 'relu', 'drop_rate': 0.2, 'batch_size': 32,
                              'random_seed': 201809, 'actincenter': "tanh", 'init': "glorot_uniform",
                              'use_earlyStop': True, 'suffix': '_'+modality, 'pretrain_stacks': True}
        default_sae_kwargs.update(sae_kwargs)

        if (self.logger is True) or (save_encoder is True):
            file_dir = tools.create_folder(self.save_dir, modality)

        if self.logger is True:
            tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=file_dir, update_freq='epoch')]

        print(sae_kwargs)

        sae = SAE(dims=[X_train.shape[-1]] + encoding_layer_dims, **default_sae_kwargs)
        print(sae.autoencoders.summary())
        sae.fit(x=X_train, epochs=epochs, decaying_step=decaying_step, callbacks=tensorboard_callback)

        ls_mat = sae.encoder.predict(X_all_data)
        self.mdata_train.mod[modality].obsm['X_desc_ls'] = ls_mat.copy()
        self.mdata_train.mod[modality].uns['desc_ls_hparams'] = {'dims': encoding_layer_dims, 'sae_kwargs': default_sae_kwargs,
                                                                 'epochs': epochs, 'decaying_step': decaying_step,
                                                                 'data_split': train_test_split}

        if save_encoder is True:
            sae.encoder.save(os.path.join(file_dir, modality+'_desc_encoder.keras'))
            with open(os.path.join(file_dir, modality+'_desc_encoder_hparams.json'), 'w') as f:
                json.dump(self.mdata_train.mod[modality].uns['desc_ls_hparams'], f)

        return sae

    @staticmethod
    def predict_ls(adata, model, use_rep=None):
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
        adata.obs[key_added] = adata.obs[key_added].astype('category')

        return adata

    def train_joint(self, modalities=None, encoding_layer_dims=None, clustering_resolutions=[0.6, 0.8], obsm_key='X_desc_ls', **kwargs):
        if modalities is not None:
            assert isinstance(modalities, list) and set(modalities).issubset(set(self.mdata_train.mod_names))
        else:
            modalities = self.mdata_train.mod_names

        ls_mat = []
        ls_names = []
        for i, modality in enumerate(modalities):
            ls_names.append([modality+'_ls_'+str(i+1) for i in np.arange(self.mdata_train.mod[modality].obsm[obsm_key].shape[1])])
            ls_mat.append(self.mdata_train.mod[modality].obsm[obsm_key])

        self.train_mod = modalities

        self.mdata_train.obsm['joint_ls'] = np.concatenate(ls_mat, axis=1)
        file_dir = tools.create_folder(self.save_dir, '_'.join(self.train_mod)+'_joint')

        dims = [self.mdata_train.obsm['joint_ls'].shape[-1]] + encoding_layer_dims
        desc_model = DESC(self.mdata_train, use_rep='joint_ls', dims=dims, save_dir=file_dir, logger=self.logger)
        model = desc_model.train(clustering_resolutions=clustering_resolutions, **kwargs)

        print(self.mdata_train)

        return model

    def build_endtoend(self, mod_sae, cluster_model):
        all_outputs = [e.encoder.output for e in mod_sae]
        all_inputs = [e.encoder.input for e in mod_sae]
        h = tf.keras.layers.Concatenate()(all_outputs)
        h = cluster_model(h)

        model = tf.keras.Model(inputs=all_inputs, outputs=h)
        self.multimodal_model = model

        return model

    @staticmethod
    def predict_clusters_multimodal(mdata, model, use_rep=None, key_added='cluster'):
        if isinstance(model, str):
            model = tools.load_model(model)
        else:
            assert isinstance(model, tf.keras.Model), ('Model provided isnt a keras model')

        x = []
        if isinstance(use_rep, dict):
            for mod, v in use_rep.items():
                if v is not None:
                    assert v in mdata.mod[mod].obsm_keys(), (f'Key {v} not found in adata.obsm for {mod} modality')
                    x.append(mdata.mod[mod].obsm[v])
                else:
                    x.append(mdata.mod[mod].X)
        else:
            for mod in mdata.mod_names:
                x.append(mdata.mod[mod].X)

        for i, e in enumerate(model.input_shape):
            assert x[i].shape[1] == e[1], (f'Encoder input expects {e[1]} features while {x[i].shape[1]} was provided')

        q = model.predict(x, verbose=0)
        mdata.obs[key_added] = q.argmax(axis=1)
        mdata.obs[key_added] = mdata.obs[key_added].astype('category')
        return mdata