import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend
from tensorflow.python.platform import tf_logging as logging


def create_folder(path, name):
    dataloc = os.path.join(path, name)
    if os.path.isdir(dataloc):
        print("Folder already exists")
    else:
        print("Folder doesn't exist.Creating new folder!")
        os.mkdir(dataloc)

    return dataloc


def calc_ari(mdata, label_col, cluster_res, prefix='desc_'):
    cluster_cols = [prefix+str(e) for e in cluster_res]
    assert set(cluster_cols).issubset(mdata.obs.columns), (f'Column(s) {set(cluster_cols) - set(mdata.obs.columns)} not found in mdata')
    assert (label_col in mdata.obs.columns), (f'Column {label_col} not found in mdata')
    ari_pd = pd.DataFrame.from_dict({e: adjusted_rand_score(mdata.obs[label_col],
                                                            mdata.obs[cluster_cols[i]])
                                     for i, e in enumerate(cluster_res)},
                                    orient='index', columns=['ARI']).rename_axis(index=['cluster resolution'])

    return ari_pd


def calc_nmi(mdata, label_col, cluster_res, prefix='desc_'):
    cluster_cols = [prefix+str(e) for e in cluster_res]
    assert set(cluster_cols).issubset(mdata.obs.columns), (f'Column(s) {set(cluster_cols) - set(mdata.obs.columns)} not found in mdata')
    assert (label_col in mdata.obs.columns), (f'Column {label_col} not found in mdata')
    nmi_pd = pd.DataFrame.from_dict({e: normalized_mutual_info_score(mdata.obs[label_col],
                                                                     mdata.obs[cluster_cols[i]],
                                                                     average_method='geometric')
                                     for i, e in enumerate(cluster_res)},
                                    orient='index', columns=['NMI']).rename_axis(index=['cluster resolution'])

    return nmi_pd


def calc_silhouette(mdata, cluster_res, cluster_prefix='desc_', embedding_prefix='X_Embedded_z'):
    cluster_cols = [cluster_prefix+str(e) for e in cluster_res]
    embedding_cols = [embedding_prefix+str(e) for e in cluster_res]
    assert set(cluster_cols).issubset(mdata.obs.columns), (f'Column(s) {set(cluster_cols) - set(mdata.obs.columns)} not found in mdata')
    assert set(embedding_cols).issubset(mdata.obsm_keys()), (f'Column(s) {set(embedding_cols) - set(mdata.obsm_keys())} not found in mdata')
    sil_pd = pd.DataFrame.from_dict({e: silhouette_score(mdata.obsm[embedding_cols[i]],
                                                         mdata.obs[cluster_cols[i]])
                                     for i, e in enumerate(cluster_res)},
                                    orient='index', columns=['Silhouette Score']).rename_axis(index=['cluster resolution'])

    return sil_pd


def plot_clustereval(metric_df, res_col='cluster resolution', met_cols=None, ncols=5, figsize=None, **plot_kwargs):
    if met_cols is None:
        met_cols = metric_df.columns.tolist()

    ncols = min([len(met_cols), ncols])
    nrows = np.ceil(len(met_cols)/ncols).astype(int)

    if figsize is None:
        figsize = (ncols*4, nrows*4.3)

    default_plot_opts = {'marker': 'o', 'linestyle': '--', 'markerfacecolor': 'b', 'c': 'magenta'}
    default_plot_opts.update(plot_kwargs)
    print(plot_kwargs)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, layout='tight', sharey=False, sharex=False)
    if (nrows > 1) | (ncols > 1):
        axes = axes.ravel()
    else:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < len(met_cols):
            sns.lineplot(data=metric_df, x=res_col, y=met_cols[i], ax=axes[i], **default_plot_opts)
            axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=13)
            axes[i].tick_params(axis='x', labelsize=12)
            axes[i].tick_params(axis='y', labelsize=12)
            # ax.set_ylim((0, 1))
            ax.set_title(f'Clustering Evaluation ({met_cols[i]})')
            axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=13)
        else:
            axes[i].set_axis_off()

    return fig


def load_model(model_loc):
    assert os.path.exists(model_loc), ('Path doesnt exist.')

    return tf.keras.models.load_model(model_loc)


def plot_architecture(model, to_file=None, **kwargs):
    default_plot_kwargs = {'show_shapes': True, 'show_dtype': True, 'show_layer_names': True,
                           'rankdir': 'TB', 'expand_nested': True, 'dpi': 150, 'show_layer_activations': True,
                           'show_trainable': True}
    default_plot_kwargs.update(kwargs)
    print(kwargs)
    fig = tf.keras.utils.plot_model(model, to_file=to_file, **default_plot_kwargs)

    return fig


class CustomReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Example:

    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```

    Args:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
    """

    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 max_lr_epochs=50,
                 early_stop=True,
                 **kwargs):
        super(CustomReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')

        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            logging.warning('`epsilon` argument is deprecated and '
                            'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.max_lr_epochs = max_lr_epochs
        self.early_stop = early_stop
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning rate reduction mode %s is unknown, '
                            'fallback to auto mode.', self.mode)
            self.mode = 'auto'

        if (self.mode == 'min' or (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.inf

        self.cooldown_counter = 0
        self.wait = 0
        self.epoch_lr_counter = 1

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.learning_rate)
        current = logs.get(self.monitor)
        self.epoch_lr_counter += 1
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))
        else:
            if not self.surpass_max_epochs():
                if self.in_cooldown():
                    self.cooldown_counter -= 1
                    self.wait = 0

                if self.monitor_op(current, self.best):
                    self.best = current
                    self.wait = 0
                elif not self.in_cooldown():
                    self.wait += 1
                    if self.wait >= self.patience:
                        old_lr = backend.get_value(self.model.optimizer.learning_rate)
                        if old_lr > np.float32(self.min_lr):
                            new_lr = np.round(old_lr * self.factor, 6)
                            new_lr = max(new_lr, self.min_lr)
                            backend.set_value(self.model.optimizer.learning_rate, new_lr)
                            if self.verbose > 0:
                                print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')

                            self.cooldown_counter = self.cooldown
                            self.wait = 0
                            self.epoch_lr_counter = 1
                        elif self.early_stop is True:
                            self.model.stop_training = True
            else:
                old_lr = backend.get_value(self.model.optimizer.learning_rate)
                if old_lr > np.float32(self.min_lr):
                    new_lr = np.round(old_lr * self.factor, 6)
                    new_lr = max(new_lr, self.min_lr)
                    backend.set_value(self.model.optimizer.learning_rate, new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}')

                    self.cooldown_counter = self.cooldown
                    self.wait = 0
                    self.epoch_lr_counter = 1
                elif self.early_stop is True:
                    self.model.stop_training = True

    def in_cooldown(self):
        return self.cooldown_counter > 0

    def surpass_max_epochs(self):
        return self.max_lr_epochs-self.epoch_lr_counter < 0