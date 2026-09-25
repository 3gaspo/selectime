"""Raw forecast inputs and the validation-combined reported methods."""
import warnings
import numpy as np

UNIVARIATE = 'vanilla_univariate'
MULTIVARIATE = 'vanilla_multivariate'


def candidate_names(k_values, model='chronos2'):
    if model == 'chronos_bolt':
        return [UNIVARIATE]
    if model not in ('chronos2', 'ts_icl'):
        raise ValueError(f'Unsupported backbone: {model}')
    scope = [MULTIVARIATE] if model == 'chronos2' else []
    return [UNIVARIATE, *scope, *[f'top_k_{k}' for k in k_values]]


def control_candidates(model, k_values=()):
    """Reported selectors and fitted mixtures mapped to their raw alternative."""
    if model == 'chronos_bolt':
        return {}
    scope = ({
        'scope_selector': MULTIVARIATE,
        'scope_selector_per_variate': MULTIVARIATE,
        'scope_mix': MULTIVARIATE,
        'scope_mix_per_variate': MULTIVARIATE,
        'scope_ridge': MULTIVARIATE,
    } if model == 'chronos2' else {})
    retrieval = {f'top_k_{k}_mix': f'top_k_{k}' for k in k_values}
    return {**scope, **retrieval}


def candidate_k(method):
    return int(method.removeprefix('top_k_').removesuffix('_mix')) if method.startswith('top_k_') else 0


def selection_rank(method):
    return ({UNIVARIATE: 0, MULTIVARIATE: 1}.get(method, 2), candidate_k(method))


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
