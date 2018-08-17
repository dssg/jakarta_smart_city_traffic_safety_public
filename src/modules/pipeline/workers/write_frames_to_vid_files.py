# ============ Base imports ======================
import os
import shlex
import subprocess as sp
# ====== External package imports ================
import numpy as np
# ====== Internal package imports ================
from src.modules.pipeline.workers.pipeline_worker import PipelineWorker
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class WriteFramesToVidFiles(PipelineWorker):
    """Put frames back together into a video file either in the middle or at the end of the pipeline
    """
    def initialize(self, buffer_size, frame_key,**kwargs):
        """
        buffer = empty object to hold frames
        buffer_size = How many frames to wait for before converting to video
        buffer_fill = counter for buffer 
        part = part of video
        frame_key = frame id
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_fill = 0
        self.part = 1  # move part
        self.frame_key = frame_key

    def startup(self):
        """Startup
        """
        self.vid_info = None

    def run(self, item):
        """Waits for a specified number of frames to fill, then appends them to each other and writes a video file to a specified path
        """
        # check if this is first frame we see
        if self.vid_info is None:
            self.vid_info= item["video_info"]
            self.imsize = 3 * self.vid_info["height"] * self.vid_info["width"]  # 3 bytes per pixel
            self.base_name = self.vid_info["file_name"].split(".")[0]
        # check if this frame is from a new video
        new_vid = not item["video_info"]["file_name"] == self.vid_info["file_name"]
        # fill up buffer with frames
        if not new_vid:
            self.buffer.append(np.uint8(item[self.frame_key]).tostring())
            self.buffer_fill += 1
        # write the video if buffer is full or if new frame is from a different video
        if (self.buffer_fill == self.buffer_size) or new_vid:
            # write frames to file
            outpath = os.path.join(self.out_path, f"{self.base_name}_{self.frame_key}_model_{self.model_number}_part_{self.part}.mkv")
            self.logger.info(f"Writing video to file: {outpath}")
            commands = shlex.split(f'ffmpeg -y -f rawvideo -vcodec rawvideo '
                                   f'-s {self.vid_info["width"]}x{self.vid_info["height"]} '
                                   f'-pix_fmt rgb24 -r {self.vid_info["fps"]} '
                                   f'-i - -an -vcodec libx264 -vsync 0 -pix_fmt yuv420p {outpath}')
            p = sp.Popen(commands, stdin=sp.PIPE, bufsize=int(self.imsize))
            p.communicate(input=b''.join(self.buffer))
            # prep for next file
            self.buffer_fill = 0
            self.buffer = []
            self.part += 1
        # update info for new video
        if new_vid:
            self.vid_info = item["video_info"]
            self.base_name = self.vid_info["file_name"].split(".")[0]
            self.buffer = [np.uint8(item[self.frame_key]).tostring()]
            self.buffer_fill += 1
            self.part = 1

    def shutdown(self):
        """Send videos to outpath and shutdown
        """
        # write final frames to file
        if self.vid_info is not None:
            outpath = os.path.join(self.out_path, f"{self.base_name}_{self.frame_key}_model_{self.model_number}_part_{self.part}.mkv")
            self.logger.info(f"Writing video to file: {outpath}")
            commands = shlex.split(f'ffmpeg -y -f rawvideo -vcodec rawvideo '
                                   f'-s {self.vid_info["width"]}x{self.vid_info["height"]} '
                                   f'-pix_fmt rgb24 -r {self.vid_info["fps"]} '
                                   f'-i - -an -vcodec libx264 -vsync 0 -pix_fmt yuv420p {outpath}')
            p = sp.Popen(commands, stdin=sp.PIPE, bufsize=int(self.imsize))
            p.communicate(input=b''.join(self.buffer))
