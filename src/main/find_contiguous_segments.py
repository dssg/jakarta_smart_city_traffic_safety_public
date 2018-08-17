# ============ Base imports ======================
import os
import glob
from datetime import datetime, timedelta
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
    """loops through video subtitle files and finds contiguous chunks of video. writes results to file

    in order to be contiguous:
        subtitles must be 1 second apart
        video timestamps must be within conf.video_sampling.frame_skip_tolerance of 1 second

    every conf.video_sampling.run_length of contiguous frames, the existing chunk is "closed" and a new chunk is
    "started"
    """
    all_sub_files = [fn for fn in glob.glob(conf.dirs.subtitles + "/*.csv") if (fn[-8:-4] != 'uuid' and fn.split('/')[-1][:4]!='test')]

    # loop through files in the subtitles directory
    runs = []
    #fmt = '%H:%M:%S.%f'
    fmt2 = '%b %d %Y %I:%M:%S %p'
    #fmt3 = '%H:%M:%S'
    for fn in all_sub_files:
        vid = fn.split('/')[-1].split('.')[0]
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
            _ = subs.readline() # remove headere
            line = subs.readline()[:-1]
            _, run_start_t_vid, _, dt = line.split(",")
            _, _, date0, year0, time0, ampm0 = dt.split(" ")
            #TODO: because day of week and month are in indonesian, this is hard coded to work for May only"
            run_start_t_street = datetime.strptime(f'May {date0} {year0} {time0} {ampm0}', fmt2)
            h, m, s = run_start_t_vid.split(":")
            run_start_t_vid = timedelta(hours=int(h), minutes=int(m), seconds=int(float(s)))
            prev_t_street = run_start_t_street
            prev_t_vid = run_start_t_vid

            # loop through each line
            for line in subs.readlines():
                # get video, start times
                line = line[:-1]
                _, t_vid, _, dt = line.split(",")
                if len(dt.split(" ")) == 1:
                    continue
                _, _, date0, year0, time0, ampm0 = dt.split(" ")
                t_street = datetime.strptime(f'May {date0} {year0} {time0} {ampm0}', fmt2)
                h, m, s = t_vid.split(":")
                t_vid = timedelta(hours=int(h), minutes=int(m), seconds=int(float(s)))
                # get difference between last two subtitles
                tdelta_vid = t_vid - prev_t_vid
                tdelta_street = t_street - prev_t_street
                # run of length run_length with no gaps:
                if run == conf.video_sampling.run_length and tdelta_vid.seconds == 1 and abs(tdelta_street.seconds - 1) < conf.video_sampling.frame_skip_tolerance:
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
                if tdelta_vid.seconds != 1 or (abs(tdelta_street.seconds - 1) >= conf.video_sampling.frame_skip_tolerance):
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

    outpath = os.path.join(conf.dirs.video_samples, "contiguous_segments.csv")
    runs_df.to_csv(outpath, index=False)
    logger.info(f"Runs written to {outpath}")


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)