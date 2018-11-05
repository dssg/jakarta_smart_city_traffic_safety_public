# ============ Base imports ======================
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.pipeline.workers.read_frames_from_vid_file import ReadFramesFromVidFile
from src.modules.pipeline.workers.read_frames_from_vid_files_in_dir import ReadFramesFromVidFilesInDir
from src.modules.pipeline.workers.read_frames_from_vid import ReadFramesFromVid
from src.modules.pipeline.workers.write_frames_to_vid_files import WriteFramesToVidFiles
from src.modules.pipeline.workers.yolo3_detect import Yolo3Detect
from src.modules.pipeline.workers.lk_sparse_optical_flow import LKSparseOpticalFlow
from src.modules.pipeline.workers.compute_frame_stats import ComputeFrameStats
from src.modules.pipeline.workers.write_keys_to_files import WriteKeysToFiles
from src.modules.pipeline.workers.write_keys_to_database_table import WriteKeysToDatabaseTable
from src.modules.pipeline.workers.mean_motion_direction import MeanMotionDirection
#from src.modules.pipeline.workers.semantic_segmenter import SemanticSegmenter
# ============== Logging  ========================
import logging
from src.modules.utils.setup import IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================

"""List of workers in the pipeline
"""

workers_dict = {
    "WriteFramesToVidFiles": WriteFramesToVidFiles,
    "ReadFramesFromVidFile": ReadFramesFromVidFile,
    "ReadFramesFromVidFilesInDir": ReadFramesFromVidFilesInDir,
    "ReadFramesFromVid": ReadFramesFromVid,
    "Yolo3Detect": Yolo3Detect,
    "LKSparseOpticalFlow": LKSparseOpticalFlow,
    "ComputeFrameStats": ComputeFrameStats,
    "WriteKeysToFiles": WriteKeysToFiles,
    "WriteKeysToDatabaseTable": WriteKeysToDatabaseTable,
    "MeanMotionDirection": MeanMotionDirection,
}
