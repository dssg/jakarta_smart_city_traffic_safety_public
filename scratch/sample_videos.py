# ============ Base imports ======================
import os
# ====== External package imports ================
import pandas as pd
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf, confp = get_config()
# ======== Load Configuration Parameters =========
path = confp.dirs.subtitles
out_dir = confp.dirs.output
vid_run_path = confp.paths.video_runs
# ================================================

setup("video_sampling")

outpath = os.path.join(out_dir, "video_sample.csv")

logger.info("Loading Data")
vids = pd.read_csv(vid_run_path)
#vids = pd.read_csv(vid_run_path)

logger.info("Adding Columns")


all_cams = vids["camera"].unique()
logger.info(f"Found {len(all_cams)} cameras")

n_missing = 0

sample = []

# for each camera:
for camera in all_cams:
    # for each day of the week
    logger.info(f"Camera {camera}")
    vids_fil_cam = vids[(vids["camera"] == camera)]
    for i in range(0, 7):
        logger.info(f"-- Day {i}")
        vids_fil_day = vids_fil_cam[(vids_fil_cam["dayofweek"] == i)]
        # for each hour of the day
        for j in range(0, 24):
            logger.info(f"---- Hour {j}")
            vids_fil_hour = vids_fil_day[(vids_fil_day["hour"] == j)]
            if not vids_fil_hour.empty:
                row = vids_fil_hour.sample(1)
                sample.append(row)
            else:
                logger.info(f"No videos for {camera} day {i}, hour {j}")
                n_missing += 1
logger.info(f"Sampled {len(sample)} videos from {vids.shape[0]}. No vids available for {n_missing} combinations")

vids_sample = pd.concat(sample)

vids_sample.to_csv(outpath, index=False)
