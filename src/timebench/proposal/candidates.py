"""The simple candidate family and its past-only covariate transformations."""
import warnings
import numpy as np

UNIVARIATE = 'vanilla_univariate'
MULTIVARIATE = 'vanilla_multivariate'
SELF_AUGMENTATION = 'self_augmentation'


def candidate_names(k_values):
    return [UNIVARIATE, MULTIVARIATE, SELF_AUGMENTATION, *[f'top_k_{k}' for k in k_values]]


def candidate_k(method):
    return int(method.removeprefix('top_k_')) if method.startswith('top_k_') else 0


def selection_rank(method):
    return ({UNIVARIATE: 0, MULTIVARIATE: 1, SELF_AUGMENTATION: 2}.get(method, 3), candidate_k(method))


def self_covariates(history):
    history = np.asarray(history, dtype=np.float32)
    return np.stack((np.sqrt(np.abs(history)), np.sign(history)))


def query_scaled_sequences(query_lookback, neighbors, horizon):
    """Use lookback-only statistics to rescale both neighbor pasts and futures."""
    query = np.asarray(query_lookback, dtype=np.float32)
    values = np.asarray(neighbors, dtype=np.float32)
    past = values[:, :-horizon]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        q_mean = np.nanmean(query)
        q_std = max(float(np.nanstd(query)), 1e-8)
        n_mean = np.nanmean(past, axis=-1, keepdims=True)
        n_std = np.maximum(np.nanstd(past, axis=-1, keepdims=True), 1e-8)
    return (values - n_mean) / n_std * q_std + q_mean


def align_covariates(sequences, context_length, horizon):
    past, future = sequences[:, :-horizon], sequences[:, -horizon:]
    if past.shape[-1] < context_length:
        past = np.pad(past, ((0, 0), (context_length - past.shape[-1], 0)), constant_values=np.nan)
    return past[:, -context_length:], future
