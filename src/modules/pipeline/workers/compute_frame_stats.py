# ============ Base imports ======================
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


class ComputeFrameStats(PipelineWorker):
    """Pipeline worker which computes frame level statistics based on keys passed to the object
    """
    def initialize(self,  count_by_class, count_threshold, count_max_prob_class_only, sum_probs_by_class,
                   sum_threshold, sum_max_prob_class_only, **kwargs):
        """stores params

        :param count_by_class: boolean, if True, counts the number of objects in each class in this frame
        :param count_threshold: float, confidence level above which to count objects
        :param count_max_prob_class_only: boolean, if True, counts only the classes of maximum probability (confidence).
         If false, then for each box, counts each class which is above count_threshold
        :param sum_probs_by_class: boolean, if True, sums up class probabilities (confidences) from boxes
        :param sum_threshold: float, confidence level above which cou sum probabilities
        :param sum_max_prob_class_only: boolean, if True, sums only the classes with maximum probability (confidence).
        If false, then for each box, sums each class probability which is above sum_threshold
        :param kwargs: additional key word arguments, for future compatability and parent class parameters
        :return: None
        """
        self.count_by_class = count_by_class
        self.count_threshold = count_threshold
        self.count_max_prob_class_only = count_max_prob_class_only
        self.sum_probs_by_class = sum_probs_by_class
        self.sum_threshold = sum_threshold
        self.sum_max_prob_class_only = sum_max_prob_class_only

    def startup(self):
        """ not used in this worker

        :return: None
        """
        pass

    def run(self, item, *args, **kwargs):
        """ processes each frame, computing its frame statistics and storing results in the frame dictionary

        :param item: Dictionary of frames
        :param args:
        :param kwargs:
        :return:
        """
        boxes = np.array(item["boxes"])
        classes = np.array(item["object_classes"])
        stats = []
        stats_labels = []
        if self.count_by_class:
            counts, count_labels = self.count_objects_by_type(boxes, classes, self.count_threshold, self.count_max_prob_class_only)
            stats.extend(counts)
            stats_labels.extend(count_labels)
        if self.sum_probs_by_class:
            sums, sum_labels = self.sum_probs_by_type(boxes, classes, self.count_threshold, self.sum_max_prob_class_only)
            stats.extend(sums)
            stats_labels.extend(sum_labels)
        item["frame_stats"] = stats
        item["frame_stats_header"] = stats_labels
        self.done_with_item(item)

    def shutdown(self):
        """ not used in this worker

        :return: None
        """
        pass

    #==============================
    #= Support Functions/Classes ==
    #==============================
    @staticmethod
    def count_objects_by_type(boxes, classes, count_threshold, count_max_prob_class_only):
        """ counts the number of objects by each type

        :param boxes: numpy array; rows are boxes, columns are [xtl, ytl, xbr, ybr, objectness, class 1 prob, class 2 prob...]
        :param classes: string names for each class
        :param count_threshold: float how high does class prob have to be before we count it?
        :param count_max_prob_class_only: boolean, if True, only count the class with highest probability
        :return: (numpy array of counts, names of count fields)
        """
        counts = [0] * len(classes)
        if count_max_prob_class_only:
            for i in range(boxes.shape[0]):
                box = boxes[i, 5:]
                box_thresh = np.zeros_like(box)
                box_thresh[np.argmax(box)] = 1.0 * (np.max(box) > count_threshold)
                counts += box_thresh
        else:
            boxes_thresh = 1.0 * (boxes > count_threshold)
            counts += np.sum(boxes_thresh, 0)[5:]
        names = [f'{cls}_counts' for cls in classes]
        return counts, names

    @staticmethod
    def sum_probs_by_type(boxes, classes, sum_threshold, sum_max_prob_class_only):
        """ sums the probabilies of each type

        :param boxes: numpy array; rows are boxes, columns are [xtl, ytl, xbr, ybr, objectness, class 1 prob, class 2 prob...]
        :param classes: string names for each class
        :param sum_threshold: float how high does class prob have to be before we sum it?
        :param sum_max_prob_class_only: boolean, if True, only count the class with highest probability
        :return: (numpy array of sums, names of sum fields)
        """
        sums = [0] * len(classes)
        if sum_max_prob_class_only:
            for i in range(boxes.shape[0]):
                box = boxes[i, 5:]
                box_thresh = np.zeros_like(box)
                box_thresh[np.argmax(box)] = np.max(box) * (np.max(box) > sum_threshold)
                sums += box_thresh
        else:
            boxes_thresh = boxes * (boxes > sum_threshold)
            sums += np.sum(boxes_thresh, 0)[5:]
        names = [f'{cls}_sums' for cls in classes]
        return sums, names
