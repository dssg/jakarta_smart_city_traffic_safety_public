# ============ Base imports ======================
import os
# ====== External package imports ================
import pandas as pd
import numpy as np
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from modules.utils.config_loader import get_config
conf, confp = get_config()
# ======== Load Configuration Parameters =========
path = confp.dirs.subtitles
out_dir = confp.dirs.output
vid_run_path = confp.paths.video_runs
vid_run_path_old = confp.paths.video_runs_old
# ================================================

setup("video_sampling_new")

outpath = os.path.join(out_dir, "video_sample_new.csv")

logger.info("Loading Data")
vids = pd.read_csv(vid_run_path)
old_vids = pd.read_csv(vid_run_path_old)
#vids = pd.read_csv(vid_run_path)

logger.info("Adding Columns")


all_cams = vids["camera"].unique()
old_cams = old_vids['camera'].unique()
check_exists = np.isin(all_cams, old_cams, invert=True)
all_cams = all_cams[check_exists]
logger.info(f"Found {len(all_cams)} cameras")

#import pdb; pdb.set_trace()

n_missing = 0

sample = []

# for each camera:
for camera in all_cams:
    # for each day of the week
    logger.info(f"Camera {camera}")
    vids_fil_cam = vids[(vids["camera"] == camera)]
    #for i in range(0, 7):
    for days in ((5,6), (0,1,2,3,4)):
        logger.info(f"-- Days {days}")
        vids_fil_day = vids_fil_cam[(vids_fil_cam["dayofweek"].isin(days))]
        # for each hour of the day
        for j in range(0, 24, 2):
            logger.info(f"---- Hour {j}")
            vids_fil_hour = vids_fil_day[(vids_fil_day["hour"] == j)]
            if vids_fil_hour.empty:
                logger.info(f"No videos for {camera} days {days}, hour {j}, checking hour {j+1}")
                vids_fil_hour = vids_fil_day[(vids_fil_day["hour"] == j+1)]
            if not vids_fil_hour.empty:
                row = vids_fil_hour.sample(1)
                sample.append(row)
            else:
                logger.info(f"No videos for {camera} days {days}, hour {j}")
                n_missing += 1
logger.info(f"Sampled {len(sample)} videos from {vids.shape[0]}. No vids available for {n_missing} combinations")

vids_sample = pd.concat(sample)

vids_sample.to_csv(outpath, index=False)
