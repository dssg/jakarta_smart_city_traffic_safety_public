# ============ Base imports ======================
import os
import itertools
# ====== External package imports ================
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# ====== Internal package imports ================
from src.modules.data.database_io import DatabaseIO
from src.modules.utils.misc import run_and_catch_exceptions
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf  = get_config()
# ================================================


def interval_overlap(interval_a, interval_b):
    """Given two intervals on the real line, determine their overlap

    :param interval_a: tuple of (left, right) for first point
    :param interval_b: tuple of (left, right) for second point
    :return: float which is overlap
    """
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def iou(box1, box2):
    """Computes the intersection over union of two boxes.

    finds the intersection area of both boxes and divides by the union of both boxes.

    :param box1: tuple of (x_topleft, y_topleft, x_bottomright, y_bottomright) for first box
    :param box2: tuple of (x_topleft, y_topleft, x_bottomright, y_bottomright) for second box
    :return: float which is IOU
    """
    intersect_w = interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
    intersect_h = interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])

    intersect = intersect_w * intersect_h

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def iou_matrix(boxes1, boxes2):
    """Compute the IOU between each pair of boxes, one from boxes1 and one from boxes2

    :param boxes1: numpy array where each row is a box
    :param boxes2: numpy array where each row is a box
    :return: matrix of iou values, each row from boxes1 and each column from boxes2
    """
    ious = np.zeros((len(boxes1), len(boxes2)))
    for i in range(len(boxes1)):
        for j in range(len(boxes2)):
            ious[i, j] = iou(boxes1[i], boxes2[j])
    return ious


def iou_allframes(true_labels, pred_labels):
    """Computes IOU matrices for each frame in

    :param true_labels: pandas dataframe of true labels for each frame
    :param pred_labels: pandas dataframe of predicted labels for each frame
    :return:
    """
    ious = []
    if pred_labels.empty:
        return None
    n_frames = np.max(pred_labels['frame'])
    for frame in range(1,n_frames+1):
        pred_here = pred_labels[pred_labels['frame'] == frame]
        true_here = true_labels[true_labels['frame'] == frame]
        true_boxes = true_here[['xtl','ytl','xbr','ybr']].values
        pred_boxes = pred_here[['xtl','ytl','xbr','ybr']].values
        ious.append(iou_matrix(true_boxes,pred_boxes))
    return ious 

def direction_calcs(direction_compare):
    """Takes a dataframe with predicted and correct motion for all matched boxes, and returns an updated dataframe with angle between motions.

    :param direction_compare: dataframe
    :return: dataframe with comparisons made
    """
    direction_compare = direction_compare.transpose()
    direction_compare.columns = ['deltax_true','deltay_true','deltax_pred','deltay_pred',
                                 'xsize_true','ysize_true','xsize_pred','ysize_pred','label_matched']
    direction_compare['magnitude_true'] = np.sqrt((direction_compare['deltax_true']**2 +
                                                  direction_compare['deltay_true']**2).astype(np.float64))
    direction_compare['magnitude_pred'] = np.sqrt((direction_compare['deltax_pred']**2 + 
                                                  direction_compare['deltay_pred']**2).astype(np.float64))
    direction_compare['size_true'] = (direction_compare['xsize_true'] + direction_compare['ysize_true'])/2
    direction_compare['size_pred'] = (direction_compare['xsize_pred'] + direction_compare['ysize_pred'])/2
    direction_compare['theta'] = 180/np.pi*np.arccos(((direction_compare['deltax_true']*direction_compare['deltax_pred'] +
                                                      direction_compare['deltay_true']*direction_compare['deltay_pred'])/
                                                     (direction_compare['magnitude_pred']*direction_compare['magnitude_true'])).astype(np.float64))
#    direction_compare.loc[(direction_compare.loc[:,'magnitude_pred'] < 1e-03) & 
#                          (direction_compare.loc[:,'magnitude_true'] > 1e-03),'theta'] = 180
#    direction_compare.loc[(direction_compare.loc[:,'magnitude_pred'] > 1e-03) & 
#                          (direction_compare.loc[:,'magnitude_true'] < 1e-03),'theta'] = 180
#    direction_compare.loc[(direction_compare.loc[:,'magnitude_pred'] < 1e-03) & 
#                          (direction_compare.loc[:,'magnitude_true'] < 1e-03),'theta'] = 0
    return direction_compare

def plot_confusion_matrix(cm, x_classes, y_classes = None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, fname = 'confmat'):
    """
    This function is adapted from scikit-learn.
    It plots the confusion matrix.
    Normalization can be applied along either axis.
    """
    if y_classes is None:
        y_classes = x_classes
    if normalize == 'true':
        denom = cm.sum(axis=0)
        denom[denom == 0] = 1
        cm = cm.astype('float')/denom
    if normalize == 'predicted':
        denom = cm.sum(axis=1)[:, np.newaxis]
        denom[denom == 0] = 1
        cm = cm.astype('float')/denom

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(x_classes))
    y_tick_marks = np.arange(len(y_classes))
    plt.xticks(x_tick_marks, x_classes, rotation=45)
    plt.yticks(y_tick_marks, y_classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.tight_layout()
    plt.savefig(f'{fname}_{normalize}.png', dpi=500)
    plt.close()

class Analysis:
    def __init__(self, model, cameras=None, times=range(24), dir_name='all',
                aggregate=None, vid_ref=None):
        """ Each instance of this class will create validation plots for each video in a set of annotated videos.
        
        :param model: an integer associated with a model id in the database
        :param cameras: a subset of the cameras to select on
        :param times: a subset of times of day to select on
        :param dir_name: the directory name to save in
        :param vid_ref: if not None, we validate only one video segment. cameras, times, aggregate will be ignored.
        :param aggregate: if None, this goes through all videos selected.
                        If a list of Analysis instances, this aggregates those.
        """
        self.labels_we_test = conf.validation.labels_we_test
        self.n_labels = len(self.labels_we_test)
        self.true_labels = self.labels_we_test + conf.validation.other_labels
        self.n_true_labels = len(self.true_labels)
        self.confusion_matrix = np.zeros((self.n_labels + 1, self.n_true_labels + 1))
        
        # build short forms of object names, for plotting
        short_forms = {}
        for label in self.true_labels:
            try:
                short_label = eval(f'conf.validation.short_names.{label}')
                short_forms[label] = short_label
            except AttributeError:
                continue
        self.labels_we_test_short = [short_forms.get(label, label) for label in self.labels_we_test]
        self.true_labels_short = [short_forms.get(label, label) for label in self.true_labels]
       
        self.label_to_idx = {self.true_labels[i]:i for i in range(self.n_true_labels)}
        self.idx_to_label = {i:self.true_labels[i] for i in range(self.n_true_labels)}
        
        self.tps_global = [[] for _ in range(self.n_labels)]
        self.fps_global = [[] for _ in range(self.n_labels)]
        self.thresh_global = [[] for _ in range(self.n_labels)]
        
        self.total_true_boxes_global = pd.Series(index = self.true_labels, data = 0)
        self.direction_dfs = []
        
        self.dbio = DatabaseIO()
        self.model = model
        self.dir_name = dir_name
        self.one_vid_run = False
        
        if aggregate is None and vid_ref is None:
            video_list = self.dbio.get_annotated_video_list()
            video_list = [i[0] for i in video_list[0]]
            
            if times == 'day':
                times = range(6,18)
            if times == 'night':
                times = list(range(6))+list(range(18,24))
            
            # TODO: put this table in database and reference it here instead
            samples_path = os.path.join(conf.dirs.video_samples, "video_sample.csv")
            video_info_chunk = pd.read_csv(samples_path)
            video_info_chunk['vid_ref'] = (video_info_chunk.video+'_'+
                                           (video_info_chunk.start_t_secs+1).apply(str)+'_to_'+
                                           video_info_chunk.stop_t_secs.apply(str))
            video_info_chunk.set_index('vid_ref', inplace=True)
            
            for vid_ref in video_list:
                self.video_info = self.dbio.get_video_info('_'.join(vid_ref.split('_')[:-3])+'.mkv')
                if cameras is not None and self.video_info['site_name'] not in cameras:
                    continue
                if video_info_chunk.loc[vid_ref,'hour'] not in times:
                    continue
                logger.info(f'Starting to process the video {vid_ref}.')
                self._pr_curves_onevid(vid_ref)
                logger.info(f'Done processing {vid_ref}.')
        
        elif vid_ref is not None:
            self.one_vid_run = True
            self.video_info = self.dbio.get_video_info('_'.join(vid_ref.split('_')[:-3])+'.mkv')
            logger.info(f'Starting to process the video {vid_ref}.')
            self._pr_curves_onevid(vid_ref)
            logger.info(f'Done processing {vid_ref}.')
            return None
        
        else:
            for analysis_inst in aggregate:
                self._update_aggregate_variables(analysis_inst.fps_global, 
                                                 analysis_inst.tps_global,
                                                 analysis_inst.thresh_global,
                                                 analysis_inst.direction_dfs,
                                                 analysis_inst.total_true_boxes_global,
                                                 analysis_inst.confusion_matrix)
            
        self._pr_curves_agg()
        
    def _update_lists(self, list_yes, list_no, pred_label_idx, true_label_idx):
        """ Adds a 1 to list_yes, a 0 to list_no, and updates the confusion matrix.
        
        :param list_yes: a list to add a 1 to.
        :param list_no: a list to add a 0 to.
        :param pred_label_idx: index corresponding to the predicted label.
        :param true_label_idx: index corresponding to the true label.
        """
        list_yes[pred_label_idx].append(1)
        list_no[pred_label_idx].append(0)
        self.confusion_matrix[pred_label_idx, true_label_idx] += 1
        
    def _pr_curves_onevid(self, vid_ref):
        """ Processes boxes for one video segment in decreasing order of confidence level.
        
        :param vid_ref: a string defining the video segment.
        """
        # Load the dataframes for true labels and predicted labels
        true_labels, pred_labels = self.get_dataframes(vid_ref)
        if true_labels is None:
            return None
        
        # We will process boxes in decreasing order of class confidence. The following array contains that order.
        confidence_argsort = np.flip(pred_labels['confidence'].argsort().values, 0)
        
        # For all pairs of boxes in the same frame, compute IOUs
        logger.info('Computing IOUs...')
        ious = iou_allframes(true_labels, pred_labels)
        
        logger.info('Creating empty lists that will house comparison results...')
        total_true_boxes = true_labels['label'].value_counts()

        tps = [[] for _ in range(self.n_labels)]
        fps = [[] for _ in range(self.n_labels)]
        thresh = [[] for _ in range(self.n_labels)]
        
        # creating lists to append direction checks to. there may be more efficient ways
        deltax_true = []
        deltay_true = []
        deltax_pred = []
        deltay_pred = []
        xsize_true = []
        ysize_true = []
        xsize_pred = []
        ysize_pred = []
        label_matched = []
        
        # iterate through boxes
        n_boxes = len(confidence_argsort)
        logger.info('Starting to process boxes!')
        for nn, ii in enumerate(confidence_argsort):
            if nn%100 == 0:
                logger.info(f'Done processing {nn}/{n_boxes} boxes')
            box = pred_labels.iloc[ii]
            frame = box['frame'] - 1 #1-indexed to 0-indexed
            pred_label_idx = self.label_to_idx[box['label']]
            thresh[pred_label_idx].append(box['confidence'])
            
            # find true box with highest iou. if it's below threshold, this is a false positive
            box_iou = ious[frame][:,box['in_frame_box_index']]
            if not box_iou.size:
                self._update_lists(fps, tps, pred_label_idx, -1)
                continue
            matched_box = np.argmax(box_iou)
            if box_iou[matched_box] < conf.validation.iou_threshold:
                self._update_lists(fps, tps, pred_label_idx, -1)
                continue
            
            # if there is a match, check if the label is the same
            box_true = true_labels[((true_labels['frame'] == frame+1) & 
                                    (true_labels['in_frame_box_index'] == matched_box))]
            true_ii = box_true.index[0]
            box_true = box_true.squeeze()
            true_label_idx = self.label_to_idx[box_true['label']]
            if box_true['label'] == box['label'] and not true_labels.loc[true_ii, 'true_positive']:
                self._update_lists(tps, fps, pred_label_idx, true_label_idx)
                pred_labels.loc[ii,'true_positive'] = True
                true_labels.loc[true_ii,'true_positive'] = True
                if frame: # first frame has no motion
                    label_matched.append(True)
            else:
                self._update_lists(fps, tps, pred_label_idx, true_label_idx)
                if frame: # first frame has no motion
                    label_matched.append(False)
            
            # if there is a match, update motion lists
            if frame: # first frame has no motion
                deltax_true.append(box_true['deltax'])
                deltay_true.append(box_true['deltay'])
                deltax_pred.append(box['deltax'])
                deltay_pred.append(box['deltay'])
                xsize_true.append(box_true['xbr']-box_true['xtl'])
                ysize_true.append(box_true['ybr']-box_true['ytl'])
                xsize_pred.append(box['xbr']-box['xtl'])
                ysize_pred.append(box['ybr']-box['ytl'])
            true_labels.loc[true_ii,'found_box'] = True
        
        # add boxes that were never matched to the confusion matrix
        for _, box in true_labels.iterrows():
            if not box['found_box']:
                true_label_idx = self.label_to_idx[box['label']]
                self.confusion_matrix[-1, true_label_idx] += 1
        
        # compare directions between all boxes matched
        direction_compare = pd.DataFrame([deltax_true,deltay_true,deltax_pred,deltay_pred,
                                          xsize_true,ysize_true,xsize_pred,ysize_pred,label_matched])
        direction_compare = direction_calcs(direction_compare)
    
        logger.info('Done with processing boxes. Cleaning up.')
        
        if self.one_vid_run:
            self.true_boxes = true_labels
            self.pred_boxes = pred_labels
            self._make_detection_plots(f'model_{self.model}/{vid_ref}', tps, fps, thresh, total_true_boxes)
        else:
            self._update_aggregate_variables(fps, tps, thresh, direction_compare, total_true_boxes)
        
    def _update_aggregate_variables(self, fps, tps, thresh, direction_compare, total_true_boxes, confusion_matrix=None):
        '''Updates all aggregate lists with lists relative to a subset of the total.
        '''
        for i in range(self.n_labels):
            self.fps_global[i] += fps[i]
            self.tps_global[i] += tps[i]
            self.thresh_global[i] += thresh[i]
        
        if isinstance(direction_compare, pd.DataFrame):
            self.direction_dfs.append(direction_compare)
        elif isinstance(direction_compare, list):
            self.direction_dfs += direction_compare
        
        classes_in_this_vid = np.intersect1d(total_true_boxes.index.values,
                                            self.total_true_boxes_global.index.values)
        self.total_true_boxes_global[classes_in_this_vid] += total_true_boxes
        
        if confusion_matrix is not None:
            self.confusion_matrix += confusion_matrix
        
    def _pr_curves_agg(self):
        """ Creates precision-recall curves and confusion matrices for aggregate of all videos processed in this instance of the class.
        
        :param dir_name: a string to become the directory name.
        """
        logger.info(f'Aggregating results for {self.dir_name}...')
        labels_present = self.n_labels*[True]
        # sort lists by decreasing order of confidence threshold
        for i in range(self.n_labels):
            sorted_lists = sorted(zip(self.thresh_global[i], self.tps_global[i], self.fps_global[i]), reverse=True)
            try:
                self.thresh_global[i], self.tps_global[i], self.fps_global[i] = list(zip(*sorted_lists))
            except ValueError: # if lists are empty, above logic would not give three lists back
                self.thresh_global[i], self.tps_global[i], self.fps_global[i] = [], [], []
        
        dir_name = f'model_{self.model}/{self.dir_name}'
        plots_dir = os.path.join(conf.dirs.pr_curves, dir_name)
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        
        logger.info('Making aggregated detection validation plots.')
        np.save(f'{plots_dir}/conf_mat.npy', self.confusion_matrix)
        plot_confusion_matrix(self.confusion_matrix, self.true_labels_short + ['nothing'],
                             self.labels_we_test_short + ['not_detected'], 'true',
                             title = f'Confusion matrix (threshold {conf.validation.confidence_threshold})',
                             fname=f'{plots_dir}/conf_mat')
        plot_confusion_matrix(self.confusion_matrix, self.true_labels_short + ['nothing'],
                             self.labels_we_test_short + ['not_detected'], 'predicted',
                             title = f'Confusion matrix (threshold {conf.validation.confidence_threshold})', 
                             fname=f'{plots_dir}/conf_mat')
        
        self._make_detection_plots(dir_name)
        
        logger.info('Making aggregated motion validation plots.')
        self._make_motion_plots(dir_name)
                
    def _make_detection_plots(self, dir_name, tps=None, fps=None, thresh=None, total_true_boxes=None):
        """Makes precision-recall curves for a video or set of videos.

        :param dir_name: name of directory where the plots will be saved
        :param tps: list of lists, one for each class, containing 0 on false positives and 1 on true positives
        :param fps: list of lists, one for each class, containing 1 on false positives and 0 on true positives
        :param thresh: list of lists, one for each class, containing the confidence level corresponding to that box
        :param total_true_boxes: dictionary containing the total number of true boxes for each class
        """
        if tps is None:
            tps = self.tps_global
        if fps is None:
            fps = self.fps_global
        if thresh is None:
            thresh = self.thresh_global
        if total_true_boxes is None:
            total_true_boxes = self.total_true_boxes_global
        
        plots_dir = os.path.join(conf.dirs.pr_curves, dir_name)
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
       
        for label in self.labels_we_test:
            i = self.label_to_idx[label]
            if (label not in total_true_boxes) or (not tps[i]) or (total_true_boxes[label]==0):
                continue
            tps_label = np.cumsum(tps[i])
            fps_label = np.cumsum(fps[i])
            precision = tps_label/(tps_label+fps_label)
            recall = tps_label/total_true_boxes[label]
            plt.plot(precision, label = 'Precision')
            plt.plot(recall, label = 'Recall')
            plt.title(f'Precision-Recall plot for {label}')
            plt.xlabel(f'{label}s labeled')
            plt.ylim([-.03,1.03])
            plt.legend()
            plt.savefig(f'{plots_dir}/pr_{label}.eps', format = 'eps', dpi = 1000)
            plt.savefig(f'{plots_dir}/pr_{label}.png')
            plt.close()
            
            thresh_label = 1 - np.array(thresh[i])
            plt.plot(thresh_label, precision, label = 'Precision')
            plt.plot(thresh_label, recall, label = 'Recall')
            plt.title(f'Precision-Recall plot for {label}')
            plt.xlabel('Uncertainty (1 $-$ confidence threshold)')
            plt.ylim([-.03,1.03])
            plt.xlim([-.03,max(thresh_label)+0.03])
            plt.legend()
            plt.savefig(f'{plots_dir}/pr_thresh_{label}.eps', format = 'eps', dpi = 1000)
            plt.savefig(f'{plots_dir}/pr_thresh_{label}.png')
            plt.close()
            
    def _make_motion_plots(self, dir_name):
        """Makes precision-recall plots and histograms for motion validation on a set of videos.
        
        :param dir_name: name of directory where the plots will be saved
        """
        # calculate true and false positives, precision, recall
        directions = pd.concat(self.direction_dfs)
        if directions.empty:
            logger.info(f'No boxes matched for {self.dir_name}!')
            logger.info('No motion analysis will be made.')
            return None
        directions.sort_values('magnitude_pred',ascending=False,inplace=True)
        directions.reset_index(inplace=True)
        n_moving = np.sum(directions['magnitude_true']>0)
        directions['tps'] = np.cumsum((directions['magnitude_pred']>0) & (directions['magnitude_true']>0))
        directions['precision'] = directions['tps']/(directions.index+1)
        directions['recall'] = directions['tps']/n_moving
        
        plots_dir = os.path.join(conf.dirs.pr_curves, dir_name)
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        
        # make plots
        plt.plot(directions['magnitude_pred'], directions['precision'], label = 'Precision')
        plt.plot(directions['magnitude_pred'], directions['recall'], label = 'Recall')
        plt.title('Precision-Recall plot for motion')
        plt.xlabel('Smallest amount of motion detected (pixels)')
        plt.xlim([directions['magnitude_pred'].max()+1,-1])
        plt.ylim([-.03,1.03])
        plt.legend()
        plt.savefig(f'{plots_dir}/motion_pr.eps', format = 'eps', dpi = 1000)
        plt.savefig(f'{plots_dir}/motion_pr.png')
        plt.close()
        plt.plot(directions['magnitude_pred'], directions['precision'], label = 'Precision')
        plt.plot(directions['magnitude_pred'], directions['recall'], label = 'Recall')
        plt.title('Precision-Recall plot for motion')
        plt.xlabel('Smallest amount of motion detected (pixels)')
        plt.xlim([5,-.05])
        plt.ylim([-.03,1.03])
        plt.legend()
        plt.savefig(f'{plots_dir}/motion_pr_zoom.eps', format = 'eps', dpi = 1000)
        plt.savefig(f'{plots_dir}/motion_pr_zoom.png')
        plt.close()
        
        for min_motion in conf.validation.minimum_motion:
            thetas = directions[(directions['theta'].notnull()) & 
                   (directions['magnitude_pred'] > min_motion)].loc[:,'theta']
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.hist(thetas, bins=36)
            ax.text(0.5,0.8,'$\\bar{\\theta}$=%.1f$^o$'%np.average(thetas),transform = ax.transAxes)
            ax.set_xlabel('Angle between predicted and true motion (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of angles for predicted motion > {min_motion} pixels')
            plt.xticks(range(0,195,15))
            plt.savefig(f'{plots_dir}/hist_motion_{min_motion}.eps', format = 'eps', dpi = 1000)
            plt.savefig(f'{plots_dir}/hist_motion_{min_motion}.png')
            plt.close()
            f = plt.figure()
            ax = f.add_subplot(111)
            ax.hist(thetas, bins=36)
            ax.text(0.5,0.8,'$\\bar{\\theta}$=%.1f$^o$'%np.average(thetas),transform = ax.transAxes)
            ax.set_xlabel('Angle between predicted and true motion (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of angles for predicted motion > {min_motion} pixels')
            plt.yscale('log')
            plt.xticks(range(0,195,15))
            plt.savefig(f'{plots_dir}/hist_motion_log_{min_motion}.eps', format = 'eps', dpi = 1000)
            plt.savefig(f'{plots_dir}/hist_motion_log_{min_motion}.png')
            plt.close()
        
    def get_dataframes(self, vid_ref):
        """Loads predicted and annotated boxes from the database.

        :param vid_ref: the video segment name to be validated
        :return: two dataframes containing the true and predicted labels, respectively
        """
        logger.info('Getting the predicted boxes from the database...')
        pred_labels, cols = self.dbio.get_results_motion(self.model, vid_ref+'.mkv')
        pred_labels = pd.DataFrame(pred_labels, columns=cols)
        pred_labels.sort_values('frame_number',inplace=True)
        
        logger.info('Getting the true annotated boxes from the database...')
        true_labels, cols = self.dbio.get_video_annotations(vid_ref)
        true_labels = pd.DataFrame(true_labels, columns=cols)
        
        logger.info('Preparing the two dataframes for comparison...')
        true_labels = true_labels[['frame', 'label', 'occluded', 'interpolated', 'xtl', 'ytl', 'xbr', 'ybr', 'deltax', 'deltay']]
        true_labels.sort_values('frame',inplace=True)
        
        try:
            pred_labels['label'] = pred_labels[self.labels_we_test].idxmax(axis=1)
            pred_labels['confidence'] = pred_labels[self.labels_we_test].max(axis=1)
        except KeyError:
            logger.info('The list labels_we_test contains labels which are not being predicted by the model! Please fix the config file.')
            raise
        except TypeError:
            logger.info('All predicted boxes have null positions! Skipping this video.')
            return None, None
        
        pred_labels = pred_labels[['frame_number', 'xtl', 'ytl', 'xbr', 'ybr', 'label', 'confidence',
                                  'mean_delta_x','mean_delta_y','magnitude']].copy()
        pred_labels.rename(columns={'frame_number':'frame', 'mean_delta_x':'deltax', 'mean_delta_y':'deltay'}, inplace=True)
        pred_labels = pred_labels[pred_labels['confidence'] > conf.validation.confidence_threshold].copy()
        
        height = self.video_info['height']
        true_labels = true_labels[true_labels['ybr'] > conf.validation.disregard_region*height].copy()
        pred_labels = pred_labels[pred_labels['ybr'] > conf.validation.disregard_region*height].copy()
        #true_labels = true_labels[true_labels['occluded'] == 't']
        true_labels.reset_index(drop=True,inplace=True)
        pred_labels.reset_index(drop=True,inplace=True)
        
        # create columns in both dataframes that index boxes in each frame, as well as indicators of matched boxes
        if pred_labels.empty:
            if true_labels.empty:
                n_frames = 1
            else:
                n_frames = np.max(true_labels['frame'])
        elif true_labels.empty:    
            n_frames = np.max(pred_labels['frame'])
        else:
            n_frames = max(np.max(pred_labels['frame']), np.max(true_labels['frame']))
        
        in_frame_box_index = []
        for frame in range(1,n_frames+1):
            n_boxes = len(pred_labels[pred_labels['frame']==frame])
            in_frame_box_index += list(range(n_boxes))
        pred_labels['in_frame_box_index'] = pd.Series(in_frame_box_index)
        in_frame_box_index = []
        for frame in range(1,n_frames+1):
            n_boxes = len(true_labels[true_labels['frame']==frame])
            in_frame_box_index += list(range(n_boxes))
        true_labels['in_frame_box_index'] = pd.Series(in_frame_box_index)
        true_labels['found_box'] = False
        pred_labels['true_positive'] = False
        true_labels['true_positive'] = False
        
        return true_labels, pred_labels

def main():
    """Runs the validation functions for all videos, and then for each camera separately.

    """
    models = conf.validation.model_numbers
    dbio = DatabaseIO()
    
    camera_list = dbio.get_camera_list()
    camera_list = [i[0] for i in camera_list[0]]
    
    for model in models: 
        logger.info(f'Starting to analyze results from model {model}.')
        analysis = []
        analysis_cam = []
        analysis_time = []
        for camera in camera_list:
            for time in ['day', 'night']:
                analysis.append(Analysis(model, camera, time, dir_name=f'{camera}_{time}'))
                analysis_cam.append(camera)
                analysis_time.append(time)
        analysis = np.array(analysis)
        analysis_cam = np.array(analysis_cam)
        analysis_time = np.array(analysis_time)
        for camera in camera_list:    
            analysis_group = Analysis(model, aggregate=analysis[analysis_cam == camera],
                                     dir_name = camera)
        for time in ['day','night']:
            analysis_group = Analysis(model, aggregate=analysis[analysis_time == time],
                                     dir_name = time)
        analysis_group = Analysis(model, aggregate=analysis)

if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
