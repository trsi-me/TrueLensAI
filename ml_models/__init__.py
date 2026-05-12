# -*- coding: utf-8 -*-
from ml_models.model_loader import (
    init_models,
    start_models_loading_thread,
    get_text_detector,
    get_image_detector,
    get_video_detector,
    wait_for_models_init,
)

__all__ = [
    "init_models",
    "start_models_loading_thread",
    "get_text_detector",
    "get_image_detector",
    "get_video_detector",
    "wait_for_models_init",
]
