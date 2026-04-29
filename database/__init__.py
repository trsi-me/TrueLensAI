# -*- coding: utf-8 -*-
from database.db_handler import (
    init_db,
    save_detection,
    get_all_history,
    get_stats,
    increment_stat,
    delete_history_record,
    clear_all_history,
)

__all__ = [
    "init_db",
    "save_detection",
    "get_all_history",
    "get_stats",
    "increment_stat",
    "delete_history_record",
    "clear_all_history",
]
