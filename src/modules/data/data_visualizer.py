# ============ Base imports ======================
import os
# ====== External package imports ================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
matplotlib.use("Agg")  # because no X11 on server
from matplotlib import pyplot as plt
# ====== Internal package imports ================
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class DataVisualizer:
    """Class to perform visualization of video metadata

    """
    def __init__(self, subdir="", extension=".png"):
        """stores params and makes directory for saving figures

        :param subdir: string, directory which contains the visualizations directory should output go
        :param extension: preferred file extension for saved figures
        """
        extension = "." + extension if "." not in extension else extension
        self.save_dir = os.path.join(conf.dirs.visualizations, subdir)
        self.extension = extension
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def video_errors(self, video):
        """creates bar charts of the error counts and types for each video file

        :param video: instance of the VideoFile class to visualize
        :param figsize: tuple (height, width) of figure to be saved
        :param alpha: alpha value for individual points
        :return: None
        """
        save_path = os.path.join(self.save_dir, video.filename) + "_errors" + self.extension
        if os.path.exists(save_path):
            return

        # Prep data
        returnstatus = video.check_for_errors()
        if returnstatus != 0:
            return
        with open(video.errors_path) as f:
            errs = pd.read_csv(f, header=0)
        errs.plot(kind="bar")
        [ax.legend(loc=2) for ax in plt.gcf().axes]
        plt.savefig(save_path)
        plt.close()

    def video_subtitle_stats(self, video, figsize=(15,10), alpha=0.7):
        """ creates various plots of subtitle information from a video file

        :param video: instance of the VideoFile class to visualize
        :param figsize: tuple (height, width) of figure to be saved
        :param alpha: alpha value for individual points
        :return: None
        """
        save_path = os.path.join(self.save_dir, video.filename) + "_subtitles" + self.extension
        if os.path.exists(save_path):
            return

        # attempt to get file names
        returnstatus = video.check_for_errors()
        if returnstatus != 0:
            return

        # Prep data
        video.get_subtitles()
        with open(video.subtitles_path) as f:
            subs = pd.read_csv(f, header=0, dtype={"subtitle_number": np.float,  # int doesn't work with NAs
                                                   "subtitle": np.str}, parse_dates=["start_time", "end_time"])
        subs["sub_duration"] = subs["end_time"]-subs["start_time"]
        # TODO: this only works for the month of Mei/May
        subs["subtitle"] = pd.to_datetime(subs["subtitle"].str[4:].str.replace("Mei", "May"))

        # plot it
        fig, ax = plt.subplots(nrows=3, ncols=1)
        try:
            subs.plot(x="subtitle_number", y="start_time", figsize=figsize, linewidth=1, ax=ax[0], alpha=alpha)
            subs.plot(x="subtitle_number", y="end_time", figsize=figsize, linewidth=1, ax=ax[0], alpha=alpha)
            ax[0].yaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        except:
            pass
        try:
            subs.plot(x="subtitle_number", y="sub_duration", figsize=figsize, linewidth=1, ax=ax[1], alpha=alpha)
        except:
            pass
        try:
            subs.plot(x="subtitle_number", y="subtitle", figsize=figsize, linewidth=1, ax=ax[2], alpha=alpha)
            ax[2].yaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M:%S'))
        except:
            pass
        ax[2].set_xlabel("subtitle_number")
        [ax.legend(loc=2) for ax in plt.gcf().axes]
        plt.savefig(save_path)
        plt.close()

    def video_packet_stats(self, video, figsize=(15,10), ave_windows=(100, 200, 500), alpha=0.7):
        """ creates various plots to visualize the packet statistics for a video. Includes windows averages

        :param video: instance of the VideoFile class to visualize
        :param figsize: tuple (height, width) of figure to be saved
        :param ave_windows: includes windowed averages
        :param alpha: alpha value for individual points
        :return: None
        """
        save_path = os.path.join(self.save_dir, video.filename) + "_packet" + self.extension
        if os.path.exists(save_path):
            return
        # prep data
        video.get_packet_stats()
        with open(video.packet_stats_path) as f:
            stats = pd.read_csv(f, header=0)
        # average keyframes over windows
        stats["flags_K"] = (stats["flags"] == "K_")*1.0
        for win in ave_windows:
            stats["flags_K_ave_{}".format(win)] = stats["flags_K"].rolling(window=win).mean()
        stats.drop("flags_K", axis=1, inplace=True)
        # average packet size over windows
        for win in ave_windows:
            stats["size_ave_{}".format(win)] = stats["size"].rolling(window=win).mean()
        stats.drop("size", axis=1, inplace=True)
        stats = stats.select_dtypes(["number"])

        # plot it
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        try:
            stats[[col for col in stats.columns if "time" in col]].plot(figsize=figsize, linewidth=1, ax=ax[0], alpha=alpha)
        except:
            pass
        try:
            stats[[col for col in stats.columns if "key_frame_ave" in col]].plot(figsize=figsize, linewidth=1, ax=ax[1], alpha=alpha)
        except:
            pass
        try:
            stats[[col for col in stats.columns if "size_ave" in col]].plot(figsize=figsize, linewidth=1, ax=ax[2], alpha=alpha)
        except:
            pass
        ax[2].set_xlabel("packet_number")
        [ax.legend(loc=2) for ax in plt.gcf().axes]
        plt.savefig(save_path)
        plt.close()

    def video_frame_stats(self, video, figsize=(15,10), ave_windows=(100, 200, 500), alpha=0.7):
        """Creates various figures from the video frame statistics

        :param video: instance of the VideoFile class to visualize
        :param figsize: tuple (height, width) of figure to be saved
        :param ave_windows: includes windowed averages
        :param alpha: alpha value for individual points
        :return: None
        """
        save_path = os.path.join(self.save_dir, video.filename) + "_frames" + self.extension
        if os.path.exists(save_path):
            return
        # prep data
        video.get_frame_stats()
        with open(video.frame_stats_path) as f:
            stats = pd.read_csv(f, header=0)
        if "key_frame" not in stats.columns and "pict_type" in stats.columns:
            stats["key_frame"] = (stats["pict_type"] == "I")*1.0
        for win in ave_windows:
            stats["key_frame_ave_{}".format(win)] = stats["key_frame"].rolling(window=win).mean()
        stats.drop("key_frame", axis=1, inplace=True)
        if "pkt_size" in stats.columns:
            for win in ave_windows:
                stats["pkt_size_ave_{}".format(win)] = stats["pkt_size"].rolling(window=win).mean()
        stats.drop("pkt_size", axis=1, inplace=True)
        stats = stats.select_dtypes(["number"])

        # plot it
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        try:
            stats[[col for col in stats.columns if "time" in col]].plot(figsize=figsize, linewidth=1, ax=ax[0], alpha=alpha)
        except:
            pass
        try:
            stats[[col for col in stats.columns if "key_frame_ave" in col]].plot(figsize=figsize, linewidth=1, ax=ax[1], alpha=alpha)
        except:
            pass
        try:
            stats[[col for col in stats.columns if "pkt_size_ave" in col]].plot(figsize=figsize, linewidth=1, ax=ax[2], alpha=alpha)
        except:
            pass
        ax[2].set_xlabel("frame_number")
        [ax.legend(loc=2) for ax in plt.gcf().axes]
        plt.savefig(save_path)
        plt.close()
