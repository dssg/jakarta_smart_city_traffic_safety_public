# ============ Base imports ======================
import os
import glob
from datetime import datetime, timedelta
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
subs_path = confp.dirs.subtitles
vid_run_path = confp.paths.video_runs_old
subtitles_dirs = confp.dirs.subtitles
# ================================================

setup("meta_data_contiguous")

# threshold for run length
run_len = 15
frame_skip_tolerance = 0.01  # how close do the video times have to be between subtitles? ideally they should be 1 second apart

run_len += 0  # add buffer to run length to shave off later
all_sub_files = [fn for fn in glob.glob(subs_path + "/*.csv") if (fn[-8:-4] != 'uuid' and fn.split('/')[-1][:4]!='test')]

# loop through files in the subtitles directory
runs = []
fmt = '%H:%M:%S.%f'
fmt2 = '%b %d %Y %I:%M:%S %p'
fmt3 = '%H:%M:%S'
for fn in all_sub_files:
    vid = fn.split('/')[-1].split('.')[0]
    # subs = pd.read_csv(fn)[:-1]

    with open(fn, 'r') as subs:
        camera = vid.split('_')[0]
        if camera[-2] == '-':
            camera = list(camera)
            camera[-2] = '_'
            camera = ''.join(camera)
        else:
            camera += '_' + vid.split('_')[1]
        n_runs = 0
        run = 0
        _ = subs.readline() # remove headers
        line = subs.readline()[:-1]
        _, run_start_t_vid, _, dt = line.split(",")
        _, _, date0, year0, time0, ampm0 = dt.split(" ")
        #TODO: because day of week and month are in indonesian, this is hard coded to work for May only"
        run_start_t_street = datetime.strptime(f'May {date0} {year0} {time0} {ampm0}', fmt2)
        #run_start_t_vid = datetime.strptime(run_start_t_vid, fmt)
        h, m, s = run_start_t_vid.split(":")
        run_start_t_vid = timedelta(hours=int(h), minutes=int(m), seconds=int(float(s)))
        prev_t_street = run_start_t_street
        prev_t_vid = run_start_t_vid

        # loop through each line
        #for _, row in subs.iterrows():
        for line in subs.readlines():
            # get video, start times
            line = line[:-1]
            _, t_vid, _, dt = line.split(",")
            if len(dt.split(" ")) == 1:
                continue
            _, _, date0, year0, time0, ampm0 = dt.split(" ")
            t_street = datetime.strptime(f'May {date0} {year0} {time0} {ampm0}', fmt2)
            #t_vid = datetime.strptime(t_vid, fmt)
            h, m, s = t_vid.split(":")
            t_vid = timedelta(hours=int(h), minutes=int(m), seconds=int(float(s)))
            # get difference between last two subtitles
            tdelta_vid = t_vid - prev_t_vid
            tdelta_street = t_street - prev_t_street
            # run of length run_len with no gaps:
            if run == run_len and tdelta_vid.seconds == 1 and abs(tdelta_street.seconds - 1) < frame_skip_tolerance:
                # check difference in video time as well
                runs.append([vid, camera,
                             str(run_start_t_vid),
                             str(t_vid),
                             int(run_start_t_vid.total_seconds()),
                             int(t_vid.total_seconds()),
                             run_start_t_street.strftime(fmt2), t_street.strftime(fmt2)])
                run = 0
                n_runs += 1
            # if gap in subs or street times, restart run
            if tdelta_vid.seconds != 1 or (abs(tdelta_street.seconds - 1) >= frame_skip_tolerance):
                run = 0
            # start new run
            if run == 0:
                run_start_t_street = t_street
                run_start_t_vid = t_vid
            # update prev time
            prev_t_street = t_street
            prev_t_vid = t_vid
            run += 1
    logger.info(f"Found {n_runs} runs in {fn}")

logger.info(f"Found {len(runs)} total runs in {len(all_sub_files)} files")
logger.info(f"Converting to DataFrame and adding day/hour/dayofweek")
# dataframe of segments with no major gaps
runs_df = pd.DataFrame(runs, columns=["video", "camera", "start_t", "stop_t", "start_t_secs", "stop_t_secs",
                                      "start_t_street", "stop_t_street"])
runs_df["day"] = pd.to_datetime(runs_df["start_t_street"]).dt.day
runs_df["hour"] = pd.to_datetime(runs_df["start_t_street"]).dt.hour
runs_df["dayofweek"] = pd.to_datetime(runs_df["start_t_street"]).dt.dayofweek

runs_df.to_csv(os.path.join(vid_run_path), index=False)
logger.info(f"Runs written to {vid_run_path}")
