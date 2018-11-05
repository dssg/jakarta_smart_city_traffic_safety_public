# ============ Base imports ======================
import os
import subprocess
import glob
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.data.video_file import VideoFile
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf, confp = get_config()
# ================================================

_, confp = setup("extract_subtitle")

# import / output files
video_file_dir = confp.dirs.videos
subtitle_dir = confp.data.subtitles
output_file_name = "video_metadata2.csv"

# call extract subtitles bash script
if not os.path.exists(subtitle_dir):
    os.mkdir(subtitle_dir)
devnull = open(os.devnull, 'w')
subprocess.run(["./extract_subtitles.sh", video_file_dir, subtitle_dir])

# create metadata file
f_out = open(subtitle_dir + output_file_name, 'w')
f_out.write("file,start_time,end_time\n")
for i, f_path in enumerate(glob.glob(subtitle_dir + "*.srt")):
    start_time = None
    end_time = None
    with open(f_path) as f:
        f_name = os.path.basename(f_path).split(".")[0]
        logger.info("Collecting metadata from file {}:{}".format(i, f_name))
        for line in f.readlines():
            if "AM" in line or "PM" in line:
                if start_time is None:
                    start_time = line.replace(",", "").replace("\n", "")
                else:
                    end_time = line.replace(",", "").replace("\n", "")
        try:
            f_out.write(",".join([f_name, start_time, end_time]) + "\n")
        except TypeError:
            f_out.write(",".join([f_name, start_time, start_time]) + "\n")
f_out.close()
