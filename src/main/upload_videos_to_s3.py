# ============ Base imports ======================
import os, gc, glob
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.data.video_file import VideoFile
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
    """Uploads videos with .mkv extension to an AWS bucket.

    if another video with the same name is found in the bucket, then this compares the file sizes, and does nothing if
    they are the same size. If they are different, the file is uploaded.

    """
    for i, f_path in enumerate(glob.glob(conf.dirs.raw_videos + "*.mkv")):
        gc.collect()
        logger.info("Processing {}".format(f_path))
        vid = VideoFile(f_path)
        upload = False
        if vid.aws.s3_vid_exists(vid.basename):
            logger.info("Video already exists on aws")
            if vid.aws.s3_get_vid_size(vid.basename) == os.path.getsize((os.path.join(conf.dirs.raw_videos, vid.basename))):
                logger.info("Sizes match")
            else:
                logger.info("Sizes don't match")
                upload = True
        else:
            logger.info("Video not already on aws")
            upload = True
        if upload:
            logger.info("Uploading")
            vid.aws.vid_copy_file_to_s3(vid.path)
            logger.info("Done")
        else:
            logger.info("Not uploading")


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
