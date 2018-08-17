# ============ Base imports ======================
import os
import shlex
import subprocess as sp
from functools import partial
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


class ReadFramesFromVidFile(PipelineWorker):
    """Breaks a video into individual frames that can be processed through the pipeline
    """
    def initialize(self, path, height, width, uuid, fps, **kwargs):
        """Initialize with parameters
        
        path = file path
        vid_name = file name
        height = frame height
        width = frame width
        uuid = unqiue identidifer
        fps = frames per second
        """
        self.path = path
        self.vid_name = os.path.basename(path)
        self.height = height
        self.width = width
        self.uuid = uuid
        self.fps = fps

    def startup(self):
        """Startup
        """
        pass

    def run(self, *args, **kwargs):
        """Read frames from video and store video info
        """
        imsize = 3 * self.height * self.width  # 3 bytes per pixel
        self.logger.info(f"Reading from file: {self.path}")
        if (not os.path.exists(self.path)) or (not os.path.isfile(self.path)):
            raise FileNotFoundError(f"Not a valid video file: {self.path}")
        commands = shlex.split(f'ffmpeg -r {self.fps} -i {self.path} -f image2pipe -vsync 0 -pix_fmt rgb24 -vcodec rawvideo -')
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
                        "file_name": self.vid_name,
                        "fps": self.fps,
                        "height": self.height,
                        "width": self.width,
                    },
                    "frame_number": i,
                    "frame": frame,
                }
                self.done_with_item(item)
            except Exception as e:
                self.logger.info(f"Done reading from file: {self.path}")
                break

    def shutdown(self):
        """Shutdown
        """
        pass
