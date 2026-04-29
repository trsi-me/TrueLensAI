# -*- coding: utf-8 -*-
"""WSGI entry for production (e.g. Gunicorn on Render)."""
from app import create_app

application = create_app()
