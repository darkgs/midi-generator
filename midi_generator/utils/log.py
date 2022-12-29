"""Logger to manage a static logger variable"""
import logging


class Log:
    """
    Log keeps a single static logger variable to print all logs from everywhere
    """

    logger = logging.getLogger()

    @staticmethod
    def info(msg: str):
        """
        Print info message

        Parameters:
            msg: str - message to print

        Returns:
            None
        """
        Log.logger.info(msg)

    @staticmethod
    def warning(msg: str):
        """
        Print warning message

        Parameters:
            msg: str - message to print

        Returns:
            None
        """
        Log.logger.warning(msg)

    @staticmethod
    def error(msg: str):
        """
        Print error message

        Parameters:
            msg: str - message to print

        Returns:
            None
        """
        Log.logger.error(msg)
