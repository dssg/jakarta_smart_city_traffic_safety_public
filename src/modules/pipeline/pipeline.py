# ============ Base imports ======================
import os
import time
import json
import multiprocessing as mp
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.pipeline.workers.workers_list import workers_dict
from src.modules.data.database_io import DatabaseIO
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf  = get_config()
# ================================================


class Pipeline:
    def __init__(self, pipe_conf):
        self.pipe_conf = pipe_conf
        self.name = pipe_conf.name
        self.tasks = pipe_conf.tasks  # list of dictionaries
        self.options = pipe_conf.options
        self.tasks_dict = dict()
        self.workers_dict = dict()
        self.workers_list = list()
        self.queues = list()
        self.queue_names = list()
        self.options = pipe_conf.options
        self.dbio = DatabaseIO()
        self.pipe_conf = pipe_conf
        ### create an entry in database
        # get maximum model id in database right now
        data, fields = self.dbio.get_max_model_number()
        last_num = data[0][0] if data[0][0] is not None else 0
        self.model_number = last_num + 1
        setup(f"pipeline_{self.model_number}")
        # set model id and upload to database
        self.dbio.insert_into_table("results", "models", ['"model_number"', '"pipeline_config"'], [f"{self.model_number}", "'" + json.dumps(self.pipe_conf, sort_keys=True, default=str, separators=(",", ":")).replace("'", "''") + "'"])
        self.out_path = os.path.join(conf.dirs.output, f"{self.model_number}")
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

    def build(self, *args, **kwargs):
        self.start_time = time.time()
        for task in self.tasks:
            # create a queue of the appropriate size
            prev_task = self.tasks_dict[task["prev_task"]] if task["prev_task"] is not None else None
            input_queue = None
            if prev_task is not None:
                q_size = prev_task["output_queue_size"]
                input_queue = mp.Queue(maxsize=q_size)
                # add queue to previous task workers
                for worker in prev_task["workers"]:
                    worker.output_queues.append(input_queue)
                self.queues.append(input_queue)
                self.queue_names.append(f"{prev_task['name']}->{task['name']}")
            # create workers for this task
            WorkerClass = workers_dict[task["worker_type"]]
            task_workers = []
            task["output_queues"] = []
            for i in range(task["num_workers"]):
                task_workers.append(WorkerClass(input_queue = input_queue, pipeline_config = self.pipe_conf, start_time = self.start_time, model_number=self.model_number, out_path=self.out_path, **task))
            self.workers_dict[task["name"]] = task_workers
            #self.workers_list.extend(task_workers)
            task["workers"] = task_workers
            self.tasks_dict[task["name"]] = task

    def run(self):
        # start pipeline monitor:
        self.qm = mp.Process(target=self.monitor_pipeline, name="monitor_pipeline")
        self.qm.start()
        # start pipeline tasks:
        for task in self.tasks:
            task["jobs"] = []
            for worker in task["workers"]:
                task["jobs"].append(mp.Process(target=worker._run, name=worker.op))
                task["jobs"][-1].start()
            logger.info(f"Started task: {task['name']}")

    def wait_for_finish(self):
        for task in self.tasks:
            for worker in task["workers"]:
                if worker.input_queue is not None:
                    worker.input_queue.put("STOP")
            for job in task["jobs"]:
                job.join()
            logger.info(f"Completed Task:{task['name']}")
        self.qm.terminate()
        self.stop_time = time.time()
        logger.info(f"Time Elapsed:{self.stop_time - self.start_time}")

    def shutdown(self):
        pass

    def monitor_pipeline(self):
        logger.info("Pipeline Monitor Starting")
        sizes = [0] * len(self.queues)
        percents = [0] * len(self.queues)
        dead = [False] * len(self.queues)
        while True:
            for i, queue in enumerate(self.queues):
                try:
                    sizes[i] = queue.qsize()
                    percents[i] = sizes[i] / queue._maxsize
                except:
                    dead[i] = True
            for i in range(len(sizes)):
                if not dead[i]:
                    bars = int(round(percents[i] * self.options.queue_monitor_meter_size))
                    empty = self.options.queue_monitor_meter_size - bars
                    logger.info("[{}{}] ({}/{}) {}".format("|"*bars, " "*empty, sizes[i],
                                                           self.queues[i]._maxsize, self.queue_names[i]))
                else:
                    logger.info("[{}] {}".format("X"*self.queues[i]._maxsize, self.queue_names[i]))
            if sum(dead) == len(sizes):
                break
            time.sleep(self.options.queue_monitor_delay_seconds)
        logger.info("Pipeline Monitor Stopping")
