import json
from functools import lru_cache

import numpy as np

from main import MATERIALS, ml_predict, rate_curves_normalized, train_models, ttt_curve


def _safe_float(value):
    if not np.isfinite(value):
        return None
    return float(value)


def _series(x_values, y_values):
    out = []
    for x, y in zip(x_values, y_values):
        xf = _safe_float(x)
        yf = _safe_float(y)
        if xf is not None and yf is not None:
            out.append([xf, yf])
    return out


@lru_cache(maxsize=4)
def _models(n_avrami: int, n_virtual: int):
    return train_models(n_virtual=n_virtual, n_T=250, n_avrami=n_avrami)


def _query(request, key, default):
    if hasattr(request, "query"):
        return request.query.get(key, default)
    if hasattr(request, "args"):
        return request.args.get(key, default)
    return default


def handler(request):
    material_name = _query(request, "material", "Zinc")
    n_avrami = int(_query(request, "n_avrami", 4))
    n_virtual = int(_query(request, "n_virtual", 200))

    if material_name not in MATERIALS:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": f"Unknown material '{material_name}'"}),
        }

    if n_avrami not in (1, 2, 3, 4):
        n_avrami = 4

    if n_virtual not in (0, 200, 500, 1000):
        n_virtual = 200

    p = MATERIALS[material_name]
    models = _models(n_avrami=n_avrami, n_virtual=n_virtual)

    t_rate, i_norm, u_norm, overall = rate_curves_normalized(p)
    t_ttt, ttt_res, t_nose, time_nose = ttt_curve(p, n_avrami=n_avrami)
    ml = ml_predict(models, p, n_avrami=n_avrami)

    payload = {
        "material": material_name,
        "n_avrami": n_avrami,
        "n_virtual": n_virtual,
        "nose": {
            "physics_temperature_c": _safe_float(t_nose),
            "physics_time_s": _safe_float(time_nose),
            "ml_temperature_c": _safe_float(ml["T_nose"]),
            "ml_time_s": _safe_float(ml["t_nose"]),
        },
        "rate_curves": {
            "nucleation": _series(i_norm, t_rate),
            "growth": _series(u_norm, t_rate),
            "overall": _series(overall, t_rate),
        },
        "ttt_curves": {
            "x_1pct": _series(ttt_res[0.01], t_ttt),
            "x_50pct": _series(ttt_res[0.50], t_ttt),
            "x_99pct": _series(ttt_res[0.99], t_ttt),
            "ml_1pct": _series(
                (np.log(100.0) / np.maximum(10.0 ** ml["log_k"], 1e-300))
                ** (1.0 / n_avrami),
                ml["T"],
            ),
        },
    }

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
