# -*- coding: utf-8 -*-
import os


class Config:
    # Optional deployment: preload models via URLs (vars used by scripts/fetch_pretrained_models.py &
    # utils/fetch_pretrained_models.py — PRETRAINED_MODELS_BASE_URL or IMAGE_MODEL_DOWNLOAD_URL / … ).
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
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
    ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
    TEXT_MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "text_model.pkl")
    TFIDF_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "tfidf_vectorizer.pkl")
    IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "saved_models", "image_model.h5")
    MAX_FRAMES_PER_VIDEO = 30
