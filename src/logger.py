import logging
import os
from datetime import datetime

LOG_FILE_NAME = f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log'
LOGS_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE_NAME)

os.makedirs(LOGS_PATH, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)
