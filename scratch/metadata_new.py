# ============ Base imports ======================
import os
import glob
from datetime import datetime, timedelta
# ====== External package imports ================
import pandas as pd
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from modules.utils.config_loader import get_config
conf, confp = get_config()
# ======== Load Configuration Parameters =========
vids_path = confp.dirs.videos_new
vid_run_path = confp.paths.video_runs
# ================================================

setup("meta_data_new")

# threshold for run length
run_len = 15
frame_skip_tolerance = 0.01  # how close do the video times have to be between subtitles? ideally they should be 1 second apart

run_len += 0  # add buffer to run length to shave off later
all_vid_files = [fn for fn in glob.glob(vids_path + "/*.mp4")]

runs = []
fmt2 = '%b %d %Y %I:%M:%S %p'
for fn in all_vid_files:
    statinfo = os.stat(fn)
    file_name = fn.split('/')[-1][:-4] 
    if statinfo.st_size == 0:
        logger.info(f'File {file_name} is empty.')
        continue
    time = file_name.split('-')[-2]
    camera = file_name[:-15]
    start_t_street = datetime.fromtimestamp(int(time)).strftime(fmt2)
    stop_t_street = datetime.fromtimestamp(int(time)+15).strftime(fmt2)
    t_vid = timedelta(seconds=0)
    t_vid_stop = timedelta(seconds=15)
    runs.append([file_name, camera, str(t_vid), str(t_vid_stop), 0, 15, start_t_street, stop_t_street])

    logger.info(f"Added a run in {file_name}")

logger.info(f"Found {len(runs)} total runs")
logger.info(f"Converting to DataFrame and adding day/hour/dayofweek")
# dataframe of segments with no major gaps
runs_df = pd.DataFrame(runs, columns=["video", "camera", "start_t", "stop_t", "start_t_secs", "stop_t_secs",
                                      "start_t_street", "stop_t_street"])
runs_df["day"] = pd.to_datetime(runs_df["start_t_street"]).dt.day
runs_df["hour"] = pd.to_datetime(runs_df["start_t_street"]).dt.hour
runs_df["dayofweek"] = pd.to_datetime(runs_df["start_t_street"]).dt.dayofweek

runs_df.to_csv(os.path.join(vid_run_path), index=False)
logger.info(f"Runs written to {vid_run_path}")
