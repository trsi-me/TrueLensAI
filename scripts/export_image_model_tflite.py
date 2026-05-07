# -*- coding: utf-8 -*-
"""
Export image_model.h5 to image_model.tflite for low-RAM hosts (e.g. Render free ~512MB).

Run locally in a venv with:  pip install -r requirements.txt
Then:  python scripts/export_image_model_tflite.py

If you see DepthwiseConv2D / 'groups' errors, you likely have standalone `keras` conflicting
with TensorFlow’s bundled Keras — run:  pip uninstall keras keras-nightly -y
Then reinstall tensorflow==2.13.0 from requirements.txt.
"""
from __future__ import annotations

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.h5")
DST = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.tflite")


def _patch_depthwise_conv2d_drop_groups() -> None:
    """
    EfficientNet .h5 from TF 2.13 may serialize DepthwiseConv2D with groups=1.
    Some standalone Keras 3 builds reject that key — strip it before from_config.
    """
    import tensorflow as tf

    def _wrap(layer_cls):
        orig = layer_cls.from_config.__func__

        @classmethod
        def from_config_fixed(cls, config):
            cfg = dict(config)
            cfg.pop("groups", None)
            return orig(cls, cfg)

        layer_cls.from_config = from_config_fixed

    _wrap(tf.keras.layers.DepthwiseConv2D)
    try:
        import keras

        k_layer = getattr(keras.layers, "DepthwiseConv2D", None)
        if k_layer is not None and k_layer is not tf.keras.layers.DepthwiseConv2D:
            _wrap(k_layer)
    except ImportError:
        pass


def main() -> int:
    if not os.path.isfile(SRC):
        print("Missing:", SRC, file=sys.stderr)
        return 1

    import tensorflow as tf

    _patch_depthwise_conv2d_drop_groups()

    from tensorflow.keras.models import load_model

    model = load_model(SRC, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []
    tflite_model = converter.convert()
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "wb") as f:
        f.write(tflite_model)
    print("Wrote", DST, "(%d bytes)" % len(tflite_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
