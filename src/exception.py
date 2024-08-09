import sys
from src.logger import logging


def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = f"Error occured in Python Script name [{filename}] at line number {line_number} with error message {str(error)}"

    return error_message


class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        super().__init__(message)
        self.error_message = error_message_details(message, error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by 0")
        raise CustomException(e, sys)
