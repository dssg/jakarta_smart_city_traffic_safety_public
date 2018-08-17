# ============ Base imports ======================
import os
from glob import glob
import json
from datetime import datetime
# ====== External package imports ================
from src.modules.utils.misc import run_and_catch_exceptions
# ====== Internal package imports ================
from src.modules.data.database_io import DatabaseIO
from src.modules.data.video_file import VideoFile
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def main(testing=False):
    """Uploads video metadata to the database

    :param testing: boolean, if True, then nothing is written to the database, it just prints what commands will run.
    """
    dbio = DatabaseIO(testing=testing)

    # upload cameras to camera table
    logger.info(f"copying contents of {conf.files.cameras} to raw.cameras")
    dbio.copy_file_to_table("raw", "cameras", conf.files.cameras)

    # for every video in the database
    for vid_path in glob(os.path.join(conf.dirs.raw_videos, "*.mkv")):
        vid = VideoFile(path=vid_path, dbio=dbio)
        logger.info(f"uploading {vid.name} to database")
        try:
            vid.upload_vid_metadata_to_db("fs")
        except Exception as e:
            js = ["upload_failure", vid.name, getattr(e, 'message', repr(e))]
            dbio.insert_into_table("main", "db_failures", ["time", "description"], [f"'{str(datetime.now())}'", f"'{json.dumps(js)}'"])


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
