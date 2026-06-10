from dataclasses import replace
from models import build_model, count_parameters

def legal_heads(d_model, requested):
    for h in [requested, 8, 4, 2, 1]:
        if h <= d_model and d_model % h == 0:
            return h
    return 1

def with_width(cfg, d_model):
    return replace(cfg, d_model=d_model, n_heads=legal_heads(d_model, cfg.n_heads))

def find_width_for_params(cfg, target_params, widths=None):
    widths = widths or list(range(16, 513, 8))
    best = None
    for w in widths:
        c = with_width(cfg, w)
        try:
            n = count_parameters(build_model(c))
        except Exception:
            continue
        err = abs(n - target_params)
        if best is None or err < best[0]:
            best = (err, c, n)
    if best is None:
        raise RuntimeError('no valid width')
    return best[1], best[2]
