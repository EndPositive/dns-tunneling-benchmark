import logging
import threading

log_lock = threading.Lock()
logger = logging.getLogger(__name__)

logging.basicConfig(
    format='%(asctime)s - %(name)s:%(lineno)s:%(funcName)20s() - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    level=logging.INFO
)

def safe_log(fn, *args, **kwargs):
    with log_lock:
        kwargs['stacklevel'] = kwargs.get('stacklevel', 2)
        return fn(*args, **kwargs)


def log_stream_logs(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream_logs=True)
    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            safe_log(logger.info, line, stacklevel=3)


def log_stream(fn, *args, **kwargs):
    output_stream = fn(*args, **kwargs, stream=True)
    for _, stream_content in output_stream:
        message = stream_content.decode("utf-8").strip()
        for line in message.splitlines():
            safe_log(logger.info, line, stacklevel=3)


def return_stream(fn, *args, **kwargs) -> str:
    output_stream = fn(*args, **kwargs, stream=True)
    output = ""
    for _, stream_content in output_stream:
        output += stream_content.decode("utf-8")
    return output
