"""Native univariate Bolt adapter, aligned with Improved TIME's base checkpoint."""
from pathlib import Path
import numpy as np


class Forecaster:
    alias = 'chronos_bolt'
    supports_covariates = False
    supports_multivariate = False

    def __init__(self, weights, device, context_length=2048):
        from chronos import BaseChronosPipeline
        from timebench.pipeline.runtime_resources import log_selected_device
        log_selected_device(device, stage="forecast", model=self.alias)
        self.context_length = context_length
        self.pipeline = BaseChronosPipeline.from_pretrained(
            str(Path(weights) / 'chronos-bolt-base'), device_map=device, local_files_only=True)

    def forecast(self, histories, horizon, *, past_covariates=None, future_covariates=None):
        import torch
        if any(value is not None and any(np.asarray(row).size for row in value)
               for value in (past_covariates, future_covariates)):
            raise ValueError('Chronos-Bolt does not support covariates')
        if any(np.asarray(history).ndim != 1 for history in histories):
            raise ValueError('Chronos-Bolt requires univariate histories')
        inputs = [torch.as_tensor(np.asarray(history[-self.context_length:], dtype=np.float32))
                  for history in histories]
        with torch.inference_mode():
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs, prediction_length=horizon, quantile_levels=[0.5])
        return [np.asarray(value.detach().float().cpu().numpy(), dtype=np.float32).reshape(1, horizon)
                for value in quantiles]
