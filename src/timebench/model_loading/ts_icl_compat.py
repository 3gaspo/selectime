"""Compatibility for the long-horizon covariate rollout in TS-ICL 0.2.1."""

from types import MethodType


def patch_tsicl_covariate_rollout(pipeline):
    """Apply the upstream rollout correction without changing the checkpoint."""
    import torch

    if getattr(pipeline, "_timebench_covariate_rollout_fixed", False):
        return pipeline
    original_rollout = pipeline._rollout_f

    def corrected_rollout(
        self,
        grid,
        series_c,
        covariates,
        has_covar,
        prediction_length,
        denormalize,
        allow_auto_complete=False,
        allow_covar_forecast=False,
    ):
        if (
            prediction_length <= self.max_target_length
            or not isinstance(covariates, torch.Tensor)
        ):
            return original_rollout(
                grid=grid,
                series_c=series_c,
                covariates=covariates,
                has_covar=has_covar,
                prediction_length=prediction_length,
                denormalize=denormalize,
                allow_auto_complete=allow_auto_complete,
                allow_covar_forecast=allow_covar_forecast,
            )

        rollouts = (
            prediction_length // self.max_target_length
            + int(prediction_length % self.max_target_length > 0)
        )
        covariate_length = covariates.shape[-2]
        covariates_cover_future = covariate_length > series_c.shape[-2]
        if covariates_cover_future:
            expected = prediction_length + series_c.shape[-2]
            if covariate_length != expected:
                raise ValueError(
                    f"Covariate sequence length {covariate_length} does not match "
                    f"context+horizon length {expected}"
                )
        covariate_past = covariates[..., : series_c.shape[-2], :]
        if covariates_cover_future:
            covariate_horizon = covariates[..., series_c.shape[-2] :, :]
        else:
            covariate_horizon = torch.full(
                (*covariates.shape[:-2], prediction_length, 1),
                torch.nan,
                device=covariates.device,
                dtype=covariates.dtype,
            )

        forecasts = []
        for rollout_index in range(rollouts):
            start = rollout_index * self.max_target_length
            chunk_length = min(
                self.max_target_length, prediction_length - start
            )
            if covariates_cover_future:
                chunk_covariates = torch.cat(
                    [
                        covariate_past,
                        covariate_horizon[..., start : start + chunk_length, :],
                    ],
                    dim=-2,
                )
            else:
                chunk_covariates = covariate_past
            chunk = self._run_forward(
                grid=grid,
                series_c=series_c,
                covariates=chunk_covariates,
                has_covar=has_covar,
                prediction_length=chunk_length,
                setting="forecasting",
                denormalize=True,
                save_scaler=rollout_index == 0,
                allow_auto_complete=allow_auto_complete,
                allow_covar_forecast=allow_covar_forecast,
            )
            forecasts.append(chunk)
            series_c = torch.cat(
                [series_c, chunk.mean(dim=-1, keepdim=True)], dim=1
            )
            covariate_past = torch.cat(
                [
                    covariate_past,
                    covariate_horizon[..., start : start + chunk_length, :],
                ],
                dim=-2,
            )

        result = torch.cat(forecasts, dim=1)
        return result if denormalize else self.scaler.transform(result)

    pipeline._rollout_f = MethodType(corrected_rollout, pipeline)
    pipeline._timebench_covariate_rollout_fixed = True
    return pipeline
