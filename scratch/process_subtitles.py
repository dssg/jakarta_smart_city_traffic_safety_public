import glob
import os
# ####### Logging and config file loading ########
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# ########### Config File Loading ################
from src.modules.utils.config_loader import get_config
conf, confp = get_config()
##################################################

conf, confp = setup("process_subtitles")

# import / output files
subtitle_dir = confp.dirs.subtitles
output_file_name = "video_metadata_full.csv"

# create metadata file
f_out = open(subtitle_dir + output_file_name, 'w')
f_out.write("file,time_stamp,date_and_time\n")
for i, f_path in enumerate(glob.glob(subtitle_dir + "*.srt")):
    with open(f_path) as f:
        f_name = os.path.basename(f_path).split(".")[0]
        print("Collecting metadata from file {}:{}".format(i, f_name))
        for line in f.readlines():
            if "AM" in line or "PM" in line:
                f_out.write(line.replace(",", "").replace("\n", "") + '\n')
            elif '-->' in line:
                f_out.write(f_name + ',' + line[:8] + ',')
f_out.close()
