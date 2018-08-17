# ============ Base imports ======================
import os
# ====== External package imports ================
from src.modules.utils.misc import run_and_catch_exceptions
# ====== Internal package imports ================
from src.modules.data.database_io import DatabaseIO
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def main(testing=False):
    """ Creates schemas and tables in the Postgres database

    :param testing: if true, nothing actually gets written to database, but logs what will be run instead
    """
    # create tables
    dbio = DatabaseIO(testing=testing)
    dbio.create_all_schemas_and_tables()

if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
