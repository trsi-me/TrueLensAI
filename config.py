# -*- coding: utf-8 -*-
import os


class Config:
    # Optional deployment: preload models via URLs (scripts/utils fetch_pretrained_models.py —
    # PRETRAINED_MODELS_BASE_URL, IMAGE_MODEL_DOWNLOAD_URL, IMAGE_MODEL_TFLITE_DOWNLOAD_URL, …).
    SECRET_KEY = os.environ.get("SECRET_KEY", "truelens-ai-secret-2024")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Training data under datasate/ (see README).
    DATASET_ROOT = os.path.join(BASE_DIR, "datasate")
    WELFAKE_CSV_PATH = os.path.join(DATASET_ROOT, "WELFake_Dataset.csv")
    IMAGE_ARCHIVE_DIR = os.path.join(DATASET_ROOT, "archive")
    IMAGE_TRAIN_DIR = os.path.join(IMAGE_ARCHIVE_DIR, "train")
    IMAGE_TEST_DIR = os.path.join(IMAGE_ARCHIVE_DIR, "test")
    DATABASE_PATH = os.path.join(BASE_DIR, "database", "truelens.db")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "assets", "uploads")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    MAX_IMAGE_UPLOAD = 20 * 1024 * 1024
    MAX_VIDEO_UPLOAD = 100 * 1024 * 1024
    # Raster / common formats Pillow can open (RGB path in image_detector). HEIC/HEIF may need system libheif / pillow-heif on some hosts.
    ALLOWED_IMAGE_EXTENSIONS = frozenset(
        {
            "png",
            "jpg",
            "jpeg",
            "jpe",
            "jfif",
            "gif",
            "webp",
            "bmp",
            "dib",
            "tiff",
            "tif",
            "ico",
            "ppm",
            "pgm",
            "pbm",
            "pnm",
            "xpm",
            "pcx",
            "tga",
            "psd",
            "jp2",
            "j2k",
            "jpc",
            "heic",
            "heif",
            "dds",
            "sgi",
            "bw",
            "rgb",
            "rgba",
            "ras",
            "sun",
            "im",
            "spider",
            "msp",
            "xbm",
        }
    )
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
    TEXT_MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "text_model.pkl")
    TFIDF_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "tfidf_vectorizer.pkl")
    IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.h5")
    # Same weights as .h5 but runnable with tflite_runtime only (avoids loading full TensorFlow).
    IMAGE_MODEL_TFLITE_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.tflite")
    MAX_FRAMES_PER_VIDEO = 30
    # When model confidence is in this band (inclusive), label is shown as "Uncertain".
    UNCERTAIN_CONFIDENCE_LOW = float(os.environ.get("TRUELENS_UNCERTAIN_LOW", "0.42"))
    UNCERTAIN_CONFIDENCE_HIGH = float(os.environ.get("TRUELENS_UNCERTAIN_HIGH", "0.58"))
