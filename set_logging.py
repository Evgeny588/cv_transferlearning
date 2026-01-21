import logging

from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent


def setup_logging(
        log_path = Path(root_dir) / 'logs/',
        filename = 'app_logs.log',
        root_level = logging.DEBUG,
        file_level = logging.INFO,
        stream_level = logging.WARNING,
        file_mode = 'w'
):  
    log_path.mkdir(parents = True, exist_ok = True)
    filepath = Path(log_path) / filename
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    formatter = logging.Formatter(
        fmt = '%(asctime)s | %(name)s | %(message)s'
    )

    # File logger
    file_logger = logging.FileHandler(
        filename = filepath,
        encoding = 'utf-8',
        mode = file_mode
    )
    file_logger.setLevel(
        file_level
    )
    file_logger.setFormatter(
        formatter
    )

    # Stream logger
    stream_logger = logging.StreamHandler()
    stream_logger.setLevel(
        stream_level
    )
    stream_logger.setFormatter(
        formatter
    )

    if not root_logger.handlers:
        root_logger.addHandler(file_logger)
        root_logger.addHandler(stream_logger)

    
