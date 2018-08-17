# ============ Base imports ======================
import os
import re
import shlex
import subprocess as sp
from functools import partial
# ====== External package imports ================
import numpy as np
# ====== Internal package imports ================
from src.modules.pipeline.workers.pipeline_worker import PipelineWorker
from src.modules.data.database_io import DatabaseIO
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


class ReadFramesFromVidFilesInDir(PipelineWorker):
    """Breaks a video into individual frames that can be processed through the pipeline
    """
    def initialize(self, vid_dir, file_regex, **kwargs):
        """Initialize with parameters
        
        vid_dir = directory path
        file_regex = regular expression to get all files
        dbio = create database
        """
        self.vid_dir = vid_dir
        self.file_regex = file_regex
        self.dbio = DatabaseIO()

    def startup(self):
        """Startup
        """
        pass

    def run(self, *args, **kwargs):
        """For each video in a folder, break into individual frames and pass to pipeline"""
        vid_files = [f for f in os.listdir(self.vid_dir) if re.search(self.file_regex, f)]
        self.logger.info(f"Found {len(vid_files)} files matching regex:{self.file_regex}")
        nl = "\n"
        self.logger.debug(f"Those files are: {nl.join(vid_files)}")
        for i, vid_file in enumerate(vid_files):
            info_dict = self.dbio.get_video_info(vid_file)
            if info_dict is None:  # this is a video chunk, which isn't in the database, so hack it to look like the video it came from
                #TODO: put video chunk metadata in the database so this hack isn't necessary
                substring = vid_file[:vid_file.find("_", vid_file.find("part")+1)] + ".%"
                info_dict = self.dbio.get_video_info(substring)
            if info_dict is None:
                self.logger.error(f"Cannot get video info for: {vid_file}, skipping")
                continue
            self.height = info_dict["height"]
            self.width = info_dict["width"]
            self.path = info_dict["file_path"]
            self.fps = info_dict["fps"]
            self.uuid = info_dict["id"]
            path = os.path.join(self.vid_dir, vid_file)
            imsize = 3 * self.height * self.width  # 3 bytes per pixel
            self.logger.info(f"Reading from file {i} of {len(vid_files)}: {path}, height:{self.height}, width:{self.width}, fps:{self.fps}, uuid:{self.uuid}")
            commands = shlex.split(f'ffmpeg -r {self.fps} -i {path} -f image2pipe -pix_fmt rgb24 -vsync 0 -vcodec rawvideo -')
            p = sp.Popen(commands, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=int(imsize))
            i = 0
            for raw_frame in iter(partial(p.stdout.read, imsize), ''):
                i += 1
                try:
                    frame = np.fromstring(raw_frame, dtype='uint8').reshape((self.height, self.width, 3))
                    item = {
                        "ops": [],
                        "video_info": {
                            "id": self.uuid,
                            "file_name": vid_file,
                            "fps": self.fps,
                            "height": self.height,
                            "width": self.width,
                        },
                        "frame_number": i,
                        "frame": frame,
                    }
                    self.done_with_item(item)
                except Exception as e:
                    self.logger.info(f"Done reading from file: {path}")
                    break
        self.logger.info(f"Done reading all files in directory: {self.vid_dir}")

    def shutdown(self):
        """Shutdown
        """
        pass
