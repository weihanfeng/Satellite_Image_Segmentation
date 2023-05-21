import logging
import logging.config
import os
import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="./conf/base/logging.yaml", default_level=logging.INFO
):
    """Set up configuration for logging utilities.

    Args:
        logging_config_path : str, optional
            Path to YAML file containing configuration for Python logger,
            by default "./config/logging_config.yaml"
        default_level : logging object, optional, by default logging.INFO
    """

    try:
        with open(logging_config_path, "rt") as file:
            log_config = yaml.safe_load(file.read())
        logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")


def get_num_files(path):
    """Get number of files in extracted folder"""
    return len(os.listdir(path))
