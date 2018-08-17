# ============ Base imports ======================
import os
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


class ReadFramesFromVid(PipelineWorker):
    def initialize(self, file_name, **kwargs):
        self.file_name = file_name
        self.dbio = DatabaseIO()

    def startup(self):
        # need height, width, path, uuid, fps
        info_dict = self.dbio.get_video_info(self.file_name)
        self.height = info_dict["height"]
        self.width = info_dict["width"]
        self.path = info_dict["file_path"]
        self.fps = info_dict["fps"]
        self.uuid = info_dict["id"]
        self.vid_info = info_dict

    def run(self, *args, **kwargs):
        imsize = 3 * self.height * self.width  # 3 bytes per pixel
        self.logger.info(f"Reading from file: {self.path}")
        if (not os.path.exists(self.path)) or (not os.path.isfile(self.path)):
            raise FileNotFoundError(f"Not a valid video file: {self.path}")
        commands = shlex.split(f'ffmpeg -r {self.fps} -i {self.path} -f image2pipe -pix_fmt rgb24 vsync 0 -vcodec rawvideo -')
        p = sp.Popen(commands, stdout=sp.PIPE, stderr=sp.DEVNULL, bufsize=int(imsize))
        i = 0
        for raw_frame in iter(partial(p.stdout.read, imsize), ''):
            i += 1
            try:
                frame = np.fromstring(raw_frame, dtype='uint8').reshape((self.height, self.width, 3))
                item = {
                    "ops": [],
                    "frame_number": i,
                    "frame": frame,
                    "video_info": self.vid_info,
                }
                self.done_with_item(item)
            except Exception as e:
                self.logger.info(f"Done reading from file: {self.path}")
                break

    def shutdown(self):
        pass
