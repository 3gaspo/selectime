"""Chronos-2's native tensor/dictionary interface, aligned with Improved TIME."""
from pathlib import Path
import numpy as np


class Forecaster:
    alias = 'chronos2'
    supports_covariates = True
    supports_multivariate = True

    def __init__(self, weights, device, context_length=8192):
        from chronos import BaseChronosPipeline

        self.context_length = context_length
        self.pipeline = BaseChronosPipeline.from_pretrained(str(Path(weights) / self.alias),
                                                          device_map=device, local_files_only=True)

    def forecast(self, histories, horizon, *, past_covariates=None, future_covariates=None):
        import torch

        if future_covariates is not None and past_covariates is None:
            raise ValueError('Future covariates require matching history channels')
        inputs = []
        for row, history in enumerate(histories):
            target = np.asarray(history, dtype=np.float32)[..., -self.context_length:]
            value = torch.as_tensor(target)
            if past_covariates is not None:
                past = np.asarray(past_covariates[row], dtype=np.float32)
                if past.shape[-1] != target.shape[-1]:
                    raise ValueError('Covariate history length must match target history')
                value = {'target': value, 'past_covariates': {
                    f'covariate_{channel}': torch.as_tensor(series) for channel, series in enumerate(past)}}
                if future_covariates is not None:
                    future = np.asarray(future_covariates[row], dtype=np.float32)
                    if future.shape != (len(past), horizon):
                        raise ValueError('Covariate future must have shape (K, horizon)')
                    value['future_covariates'] = {f'covariate_{channel}': torch.as_tensor(series)
                                                  for channel, series in enumerate(future)}
            inputs.append(value)
        with torch.inference_mode():
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=inputs, prediction_length=horizon, quantile_levels=[0.5],
                context_length=self.context_length, cross_learning=False, limit_prediction_length=False)
        if len(quantiles) != len(histories):
            raise ValueError('Chronos-2 returned a different number of forecasts than input queries')
        results = []
        for history, value in zip(histories, quantiles):
            values = value.detach().float().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
            channels = np.asarray(history).shape[0] if np.asarray(history).ndim == 2 else 1
            if values.shape != (channels, horizon, 1):
                raise ValueError(f'Unexpected Chronos-2 quantile shape {values.shape}')
            results.append(np.asarray(values[..., 0], dtype=np.float32))
        return results
