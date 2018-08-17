# ============ Base imports ======================
import os
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.pipeline.pipeline import Pipeline
from src.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def main():
    """Runs a pipeline model, as specified in the configuration parameters

    """
    # make pipeline
    pl = Pipeline(conf.pipeline)
    pl.build()
    pl.run()
    pl.wait_for_finish()
    pl.shutdown()


if __name__ == "__main__":
    run_and_catch_exceptions(logger, main)
