def load_forecaster(alias, weights, device, context_length):
    if alias == 'chronos2':
        from .chronos2 import Forecaster
    elif alias == 'chronos_bolt':
        from .chronos_bolt import Forecaster
    elif alias == 'ts_icl':
        from .ts_icl import Forecaster
    else:
        raise ValueError(f'Unsupported backbone: {alias}')
    return Forecaster(weights, device, context_length)
