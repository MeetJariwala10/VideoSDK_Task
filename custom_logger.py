"""Lightweight structured logger for the project.

This module configures a file-based JSON logger using `structlog` so that
our custom events (like `RAG HIT`/`RAG MISS`) are captured cleanly in the
`logs/` directory. It also silences noisy third-party libraries.
"""

import os
import logging
from datetime import datetime
import structlog

class CustomLogger:
    def __init__(self, log_dir: str = "logs"):
        """Create a new logger that writes to a timestamped file in `log_dir`."""
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Timestamped log file
        log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name: str = __file__):
        """Return a structlog logger that writes JSON lines to our log file."""
        logger_name = os.path.basename(name)

        # Only file output; no console
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Create a dedicated logger instead of touching root
        logging_logger = logging.getLogger(logger_name)
        logging_logger.setLevel(logging.INFO)
        logging_logger.propagate = False  # <-- Prevent logs from going to root
        logging_logger.addHandler(file_handler)

        # Silence noisy libraries (Mistral, Chroma, Cerebras)
        for lib in ["httpx", "chroma", "aiohttp", "urllib3", "langchain"]:
            logging.getLogger(lib).setLevel(logging.WARNING)

        # Structlog setup
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True
        )

        return structlog.get_logger(logger_name)


# --- Example (manual) usage ---
if __name__ == "__main__":
    pass
    # logger = CustomLogger().get_logger(__file__)
    # logger.info("User uploaded a file", user_id=123, filename="report.pdf")
    # logger.error("Failed to process PDF", error="File not found", user_id=123)
