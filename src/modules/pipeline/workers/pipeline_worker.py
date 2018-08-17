# ============ Base imports ======================
from abc import ABC, abstractmethod
from time import time
import multiprocessing as mp
import traceback
import sys
from types import SimpleNamespace
from functools import partial
# ====== External package imports ================
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

class PipelineWorker(ABC):
    """ Parent Class for all pipeline workers
    """
    def __init__(self, name, input_queue, output_queues, model_number, pipeline_config, start_time, out_path, *args, **kwargs):
        self.name = name
        self.op = f"{self.__class__.__name__}:{self.name}"
        self.logger = SimpleNamespace()
        for name, lvl in zip(["critical", "error", "warning", "info", "debug"], [50, 40, 30, 20, 10]):
            self.logger.__setattr__(name, partial(self.log, level=lvl, prefix=f"M({model_number}){self.op}"))
        self.input_queue = input_queue
        self.output_queues = output_queues
        self.model_number = model_number
        self.pipeline_config = pipeline_config
        self.start_time = start_time
        self.out_path = out_path
        self.logger.info("Starting Worker")
        self.initialize(*args, **kwargs)


    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def startup(self):
        pass

    # it's up to the run function to place a frame or frames in the output queue
    @abstractmethod
    def run(self, item=None):
        pass

    @abstractmethod
    def shutdown(self, *args, **kwargs):
        pass

    def log(self, msg, level, prefix):
        logger.log(level, f"{prefix}|" + "{0:.2f}".format(time()-self.start_time) + f"|{msg}")

    def _run(self):
        self.logger.info("Starting Up")
        self.startup()
        self.logger.info("Running")
        if self.input_queue is None:
            # this worker is a source, just run
            run_and_catch_exceptions(self.logger, self.run)
            run_and_catch_exceptions(self.logger, self.shutdown)
        else:
            # this worker is monitoring a queue, get each item and call run with it
            for item in iter(self.input_queue.get, 'STOP'):
                run_and_catch_exceptions(self.logger, self.run, item)
            run_and_catch_exceptions(self.logger, self.shutdown)
        self.logger.info("Done running")

    # place item in output queue
    def done_with_item(self, item):
        item["op"] = self.op
        for q in self.output_queues:
            q.put(item)

    # place multiple items in output queue
    def done_with_items(self, items):
        for item in items:
            self.done_with_item(item)

    # done working
    def _shutdown(self, *args, **kwargs):
        self.logger.info("Shutting Down")
        self.shutdown(*args, **kwargs)

