# ============ Base imports ======================
import os
import subprocess as sp
import shlex
# ====== External package imports ================
# ====== Internal package imports ================
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
    """runs bash script which extracts video segments as specified in the video_sample.csv file

    """
    script_path = os.path.join(conf.dirs.scripts, "extract_video_segments.sh")
    video_dir = conf.dirs.raw_videos
    samples_file = os.path.join(conf.dirs.video_samples, "video_sample.csv")
    video_out_dir = conf.dirs.video_samples
    sp.run(shlex.split(f"{script_path} {video_dir} {samples_file} {video_out_dir}"))


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
