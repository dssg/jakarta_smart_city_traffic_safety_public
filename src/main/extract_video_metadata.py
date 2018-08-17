# ============ Base imports ======================
import os, glob, gc
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
    """Loop through each raw video and extract its subtitles, frame statistics, and packet statistics into files

    """
    for i, f_path in enumerate(glob.glob(conf.dirs.raw_videos + "*.mkv")):
        gc.collect()
        logger.info("Processing {}".format(f_path))
        vid = VideoFile(f_path)
        logger.info("Extracting subtitles")
        vid.extract_subtitles()
        logger.info("Extracting Frame statistics")
        vid.extract_frame_stats()
        logger.info("Extracting Packet statistics")
        vid.extract_packet_stats()
        logger.info(f"Done with {f_path}")
    logger.info(f"Done!")


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)