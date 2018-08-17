# ============ Base imports ======================
import os
from collections import Iterable
# ====== External package imports ================
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


class WriteKeysToFiles(PipelineWorker):
    def initialize(self, path_base, keys, keys_headers, flush_buffer_size, **kwargs):
        self.path_base = path_base
        self.keys = keys
        self.keys_headers = keys_headers
        self.flush_buffer_size = flush_buffer_size

    def startup(self):
        self.vid_info = None
        self.output_files = []

    def run(self, item, *args, **kwargs):
        # check if first time through
        if self.vid_info is None:
            self.vid_info = item["video_info"]
            self.open_files()
        # check if new video and if so start a new file
        new_vid = not item["video_info"]["file_name"] == self.vid_info["file_name"]
        if new_vid:
            self.close_files()
            self.vid_info = item["video_info"]
            self.open_files()
        # for each key, write to its file
        for i in range(len(self.keys)):
            f = self.output_files[i]
            key = self.keys[i]
            key_header = self.keys_headers[i] if self.keys_headers is not None else False
            header_written = self.headers_written[i]
            try:
                data = item[key]
                if key_header and (not header_written):
                    self.logger.info(f"Writing headers for key: {key}")
                    headers = item[key_header]
                    f.write(f"frame_number,{','.join([str(thing) for thing in headers])}\n")
                    self.headers_written[i] = True
                string = self.make_string(item["frame_number"], data)
                if string is not None:
                    f.write(string)
            except Exception as e:
                self.logger.error(f"Could not write data for key:{key}, item:{item['frame_number']}")
                raise e


    def shutdown(self):
        self.close_files()

    def make_string(self, prefix, data):
        # if the data are not a list, just convert to string
        if not isinstance(data, Iterable):
            return f"{prefix},{str(data)}\n"
        datal = [datum for datum in data]
        if len(datal) == 0:
            return None
        if isinstance(datal[0], Iterable):
            nl = "\n"
            return nl.join([f"{prefix},{','.join([str(datum2) for datum2 in datum])}" for datum in datal])+nl
        else:
            return f"{prefix},{','.join([str(datum) for datum in datal])}\n"

    def open_files(self):
        for key in self.keys:
            fn = self.vid_info["file_name"].split(".")[0]
            path = os.path.join(self.path_base, f"{fn}_{key}.csv")
            self.logger.info(f"Opening file for writing:{path}")
            if not os.path.isdir(self.path_base):
                os.makedirs(self.path_base)
            if self.flush_buffer_size is None:
                self.output_files.append(open(path, 'w'))
            else:
                self.output_files.append(open(path, 'w', self.flush_buffer_size))
            self.headers_written = [False] * len(self.keys)

    def close_files(self):
        for f in self.output_files:
            self.logger.info(f"Closing file:{f.name}")
            f.close()
        self.output_files = []
