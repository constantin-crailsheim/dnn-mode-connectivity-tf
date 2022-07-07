import logging


def configure_loggers(log_level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d (%(levelname)s) [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("mode_connectivity")
    logger.setLevel(log_level)
