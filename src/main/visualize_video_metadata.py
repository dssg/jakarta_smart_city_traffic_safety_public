# ============ Base imports ======================
import os
from datetime import datetime
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.data.video_file import VideoFile
from src.modules.data.data_visualizer import DataVisualizer
from src.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from src.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def main():
    """Creates visualizations of video metadata

    Including errirs, frame_statistics, packet statistics, and subtitle statistics.
    """
    # create a data visualizer
    dv = DataVisualizer(conf, "pdf")

    # get a video
    for filename in os.listdir(conf.dirs.raw_videos):
        # make sure it's one of the accepted container types
        if not any([c_type in filename for c_type in conf.visualiation.vid_containers]):
            continue
        print("{}: Visualizing {}".format(datetime.now(), filename))

        v = VideoFile(path=os.path.join(conf.dirs.out_vid_vis_subdir, filename))

        # visualize frame statistics
        dv.video_errors(v)
        dv.video_frame_stats(v)
        dv.video_packet_stats(v)
        dv.video_subtitle_stats(v)


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
