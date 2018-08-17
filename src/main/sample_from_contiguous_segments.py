# ============ Base imports ======================
import os
# ====== External package imports ================
import pandas as pd
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
    """generates random samples from the contiguous segments which have been identified in contiguous segments.

    Specifically, this looks at each camera, at weekends separately from weekdays, and at even hours only (unless there
    are no continuous segments from even hours, then the next odd hour is used.

    For each camera, weekend/weekday, hour combination, a contiguous segment is selected at random
    results are saved to a file
    """
    if not os.path.isdir(conf.dirs.video_samples):
        os.makedirs(conf.dirs.video_samples)
    outpath = os.path.join(conf.dirs.video_samples, "video_sample.csv")

    logger.info("Loading Data")
    inpath = os.path.join(conf.dirs.video_samples, "contiguous_segments.csv")
    vids = pd.read_csv(inpath)

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
        for days in ((5,6), (0,1,2,3,4)):
            logger.info(f"-- Days {days}")
            vids_fil_day = vids_fil_cam[(vids_fil_cam["dayofweek"].isin(days))]
            # for each hour of the day
            for j in range(0, 24):
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


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
