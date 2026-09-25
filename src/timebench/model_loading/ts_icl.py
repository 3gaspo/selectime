"""Native TS-ICL adapter; target channels are independent, covariates mix channels.

Source-adapted from Evaluating TSFMs' experiments/ts_icl.py (5074a23), using
tsicl 0.2.1's native past-only/fully observed covars and median quantiles.
"""
from pathlib import Path
import numpy as np


class Forecaster:
    alias = 'ts_icl'
    supports_covariates = True
    supports_multivariate = False

    def __init__(self, weights, device, context_length=2048):
        from tsicl import TSICL
        from timebench.pipeline.runtime_resources import log_selected_device
        log_selected_device(device, stage="forecast", model=self.alias)
        self.context_length, self.device = context_length, device
        self.pipeline = TSICL(model_path=str(Path(weights) / 'tsicl/tsicl-v1.ckpt'),
                              allow_auto_download=False)

    def forecast(self, histories, horizon, *, past_covariates=None, future_covariates=None):
        import torch
        if future_covariates is not None and past_covariates is None:
            raise ValueError('Future covariates require matching history channels')
        contexts, covariates = [], []
        for row, history in enumerate(histories):
            target = np.asarray(history, dtype=np.float32)[-self.context_length:]
            if target.ndim != 1:
                raise ValueError('TS-ICL requires independent univariate targets')
            contexts.append(torch.as_tensor(target[:, None]))
            if past_covariates is not None:
                past = np.asarray(past_covariates[row], dtype=np.float32)
                if past.shape[-1] != len(target):
                    raise ValueError('Covariate history length must match target history')
                if future_covariates is not None:
                    future = np.asarray(future_covariates[row], dtype=np.float32)
                    if future.shape != (len(past), horizon):
                        raise ValueError('Covariate future must have shape (K, horizon)')
                    past = np.concatenate((past, future), axis=-1)
                covariates.append(torch.as_tensor(past.T))
        groups = {}
        for index, context in enumerate(contexts):
            key = (tuple(context.shape), tuple(covariates[index].shape) if covariates else None)
            groups.setdefault(key, []).append(index)
        result = [None] * len(contexts)
        for indices in groups.values():
            inputs = torch.stack([contexts[index] for index in indices])
            covars = torch.stack([covariates[index] for index in indices]) if covariates else None
            with torch.inference_mode():
                _, quantiles = self.pipeline.forecast(
                    inputs=inputs, covars=covars, prediction_length=horizon, batch_size=len(indices),
                    quantile_levels=[0.5], context_length=inputs.shape[1], device=torch.device(self.device),
                    denormalize=True, squeeze_output=False, allow_auto_complete=False,
                    allow_covar_forecast=False)
            values = quantiles.detach().float().cpu().numpy()
            if values.shape != (len(indices), 1, horizon, 1):
                raise ValueError(f'Unexpected TS-ICL quantile shape {values.shape}')
            for local, index in enumerate(indices):
                result[index] = np.asarray(values[local, ..., 0], dtype=np.float32)
        return result
