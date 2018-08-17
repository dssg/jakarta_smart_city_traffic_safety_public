# ============ Base imports ======================
import os
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


def update_lists(list_yes, list_no, label_idx):
    """TODO: JOAO should explain this

    :param list_yes: ???
    :param list_no: ???
    :param label_idx: ???
    """
    try:
        list_yes[label_idx].append(list_yes[label_idx][-1]+1)
        list_no[label_idx].append(list_no[label_idx][-1])
    except IndexError:
        list_yes[label_idx].append(1)
        list_no[label_idx].append(0)



def direction_calcs(direction_compare):
    """TODO: Joao should explain this

    :param direction_compare: ???
    :return: ???
    """
    direction_compare = direction_compare.transpose()
    direction_compare.columns = ['deltax_true','deltay_true','deltax_pred','deltay_pred','xsize_true','ysize_true','xsize_pred','ysize_pred']
    direction_compare['magnitude_true'] = np.sqrt(direction_compare['deltax_true']**2 + 
                                                  direction_compare['deltay_true']**2)
    direction_compare['magnitude_pred'] = np.sqrt(direction_compare['deltax_pred']**2 + 
                                                  direction_compare['deltay_pred']**2)
    direction_compare['size_true'] = (direction_compare['xsize_true'] + direction_compare['ysize_true'])/2
    direction_compare['size_pred'] = (direction_compare['xsize_pred'] + direction_compare['ysize_pred'])/2
    direction_compare['theta'] = 180/np.pi*np.arccos((direction_compare['deltax_true']*direction_compare['deltax_pred'] +
                                                      direction_compare['deltay_true']*direction_compare['deltay_pred'])/
                                                     (direction_compare['magnitude_pred']*direction_compare['magnitude_true']))
    direction_compare.loc[(direction_compare.loc[:,'magnitude_pred'] < 1e-03) & (direction_compare.loc[:,'magnitude_true'] > 1e-03),'theta'] = 180
    direction_compare.loc[(direction_compare.loc[:,'magnitude_pred'] > 1e-03) & (direction_compare.loc[:,'magnitude_true'] < 1e-03),'theta'] = 180
    direction_compare.loc[(direction_compare.loc[:,'magnitude_pred'] < 1e-03) & (direction_compare.loc[:,'magnitude_true'] < 1e-03),'theta'] = 0
    return direction_compare


class Analysis:
    def __init__(self):
        """ TODO: Joao

        """
        self.labels_we_test = conf.validation.labels_we_test
        self.n_labels = len(self.labels_we_test)
        self.true_labels = self.labels_we_test + conf.validation.other_labels
        self.n_true_labels = len(self.true_labels)
        self.confusion_matrix = np.zeros((self.n_labels + 1, self.n_true_labels + 1))
        
        self.label_to_idx = {self.true_labels[i]:i for i in range(self.n_true_labels)}
        self.idx_to_label = {i:self.true_labels[i] for i in range(self.n_true_labels)}
        
        self.tps_global = [[] for _ in range(self.n_labels)]
        self.fps_global = [[] for _ in range(self.n_labels)]
        self.thresh_global = [[] for _ in range(self.n_labels)]
        
        self.total_true_boxes_global = pd.Series(index = self.true_labels, data = 0)
        self.direction_dfs = []
        
        self.dbio = DatabaseIO()
        
    def update_lists(self, list_yes, list_no, pred_label_idx, true_label_idx):
        list_yes[pred_label_idx].append(1)
        list_no[pred_label_idx].append(0)
        self.confusion_matrix[pred_label_idx, true_label_idx] += 1
        
    def pr_curves_onevid(self, vid_ref):
        """ Load the dataframes for true labels and predicted labels

        :param vid_ref: TODO: Joao explain?
        """
        true_labels_file = conf.dirs.annotations + vid_ref + '.csv'
        pred_labels_file = conf.dirs.pred_boxes + vid_ref + '_boxes.csv'
        pred_motion_file = conf.dirs.pred_boxes + vid_ref + '_points_grouped_by_box.csv'
        self.video_info = self.dbio.get_video_info('_'.join(vid_ref.split('_')[:-3])+'.mkv')
        
        true_labels, pred_labels, motion = self.get_dataframes(true_labels_file, pred_labels_file, pred_motion_file)
        
        # We will process boxes in decreasing order of class confidence. The following array contains that order.
        confidence_argsort = np.flip(pred_labels['confidence'].argsort().values, 0)
        
        # For all pairs of boxes in the same frame, compute IOUs
        logger.info('Computing IOUs...')
        ious = iou_allframes(true_labels, pred_labels)
        
        logger.info('Some more boring setup...')
        total_true_boxes = true_labels['label'].value_counts()

        tps = [[] for _ in range(self.n_labels)]
        fps = [[] for _ in range(self.n_labels)]
        thresh = [[] for _ in range(self.n_labels)]
        
        # creating lists to append direction checks to. there may be more efficient ways
        if motion:
            deltax_true = []
            deltay_true = []
            deltax_pred = []
            deltay_pred = []
            xsize_true = []
            ysize_true = []
            xsize_pred = []
            ysize_pred = []
        
        n_boxes = len(confidence_argsort)
        logger.info('Starting to process boxes!')
        for nn, ii in enumerate(confidence_argsort):
            if nn%100 == 0:
                logger.info(f'Done processing {nn}/{n_boxes} boxes')
            box = pred_labels.iloc[ii]
            frame = box['frame'] - 1 #1-indexed to 0-indexed
            pred_label_idx = self.label_to_idx[box['label']]
            thresh[pred_label_idx].append(box['confidence'])
            
            box_iou = ious[frame][:,box['in_frame_box_index']]
            if not box_iou.size:
                self.update_lists(fps, tps, pred_label_idx, -1)
                continue
            matched_box = np.argmax(box_iou)
            if box_iou[matched_box] < conf.validation.iou_threshold:
                self.update_lists(fps, tps, pred_label_idx, -1)
                continue
                
            box_true = true_labels[((true_labels['frame'] == frame+1) & 
                                    (true_labels['in_frame_box_index'] == matched_box))]
            true_ii = box_true.index[0]
            box_true = box_true.squeeze()
            true_label_idx = self.label_to_idx[box_true['label']]
            if box_true['label'] == box['label'] and not true_labels.loc[true_ii, 'true_positive']:
                self.update_lists(tps, fps, pred_label_idx, true_label_idx)
                pred_labels.loc[ii,'true_positive'] = True
                true_labels.loc[true_ii,'true_positive'] = True
            else:
                self.update_lists(fps, tps, pred_label_idx, true_label_idx)
            if frame and motion: # first frame has no motion
                deltax_true.append(box_true['deltax'])
                deltay_true.append(box_true['deltay'])
                deltax_pred.append(box['deltax'])
                deltay_pred.append(box['deltay'])
                xsize_true.append(box_true['xbr']-box_true['xtl'])
                ysize_true.append(box_true['ybr']-box_true['ytl'])
                xsize_pred.append(box['xbr']-box['xtl'])
                ysize_pred.append(box['ybr']-box['ytl'])
            true_labels.loc[true_ii,'found_box'] = True
        
        for _, box in true_labels.iterrows():
            if not box['found_box']:
                true_label_idx = self.label_to_idx[box['label']]
                self.confusion_matrix[-1, true_label_idx] += 1
        
        if motion:
            direction_compare = pd.DataFrame([deltax_true,deltay_true,deltax_pred,deltay_pred,xsize_true,ysize_true,xsize_pred,ysize_pred])
            direction_compare = direction_calcs(direction_compare)
    
        logger.info('Done with processing boxes. Making plots.')
        
        self.make_plots(vid_ref, tps, fps, thresh, total_true_boxes)

        for i in range(self.n_labels):
            self.fps_global[i] += fps[i]
            self.tps_global[i] += tps[i]
            self.thresh_global[i] += thresh[i]
        self.direction_dfs.append(direction_compare)
        classes_in_this_vid = np.intersect1d(total_true_boxes.index.values,
                                            self.total_true_boxes_global.index.values)
        self.total_true_boxes_global[classes_in_this_vid] += total_true_boxes
        
    def pr_curves_agg(self, dir_name = 'all'):
        for i in range(self.n_labels):
            sorted_lists = sorted(zip(self.thresh_global[i], self.tps_global[i], self.fps_global[i]), reverse=True)
            self.thresh_global[i], self.tps_global[i], self.fps_global[i] = list(zip(*sorted_lists))
        
        plots_dir = os.path.join(conf.dirs.pr_curves, dir_name)
        if not os.path.isdir(plots_dir):
            os.makedirs(plots_dir)
        np.save(f'{plots_dir}/conf_mat.npy', self.confusion_matrix)
        self.make_plots(dir_name)
        
                
    def make_plots(self, dir_name, tps=None, fps=None, thresh=None, total_true_boxes=None):
        """TODO: Joao

        :param dir_name:
        :param tps:
        :param fps:
        :param thresh:
        :param total_true_boxes:
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
            if (label not in total_true_boxes) or (not tps[i]):
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
        

    def get_dataframes(self, true_labels_file, pred_labels_file, pred_motion_file=None):
        """TODO: Joao

        :param true_labels_file:
        :param pred_labels_file:
        :param pred_motion_file:
        :return:
        """
        logger.info('Reading true labels file...')
        true_labels = pd.read_csv(true_labels_file)
        logger.info('Reading predicted labels file...')
        pred_labels = pd.read_csv(pred_labels_file)
        logger.info('Reading predicted direction file...')
        if pred_motion_file is not None:
            pred_motion = pd.read_csv(pred_motion_file)
            try:
                assert(len(pred_labels) == len(pred_motion))
            except AssertionError:
                logger.info('The motion predictions do not have the same number of elements as the object predictions! Ignoring motion.')
                motion = False
        else:
            motion = False
        
        true_labels = true_labels[['frame', 'label', 'occluded', 'interpolated', 'xtl', 'ytl', 'xbr', 'ybr', 'deltax', 'deltay']]
        true_labels.sort_values('frame',inplace=True)
        
        try:
            pred_labels['label'] = pred_labels[self.labels_we_test].idxmax(axis=1)
            pred_labels['confidence'] = pred_labels[self.labels_we_test].max(axis=1)
        except KeyError:
            logger.info('The list labels_we_test contains labels which are not being predicted by the model! Please fix the config file.')
            raise
        pred_labels = pred_labels[['frame_number', 'xtl', 'ytl', 'xbr', 'ybr', 'label', 'confidence']]
        if pred_motion_file is not None:
            pred_labels = pd.concat([pred_labels, pred_motion[['mean_delta_x','mean_delta_y','magnitude']]], axis=1)
            motion = True
        pred_labels.rename(columns={'frame_number':'frame', 'mean_delta_x':'deltax', 'mean_delta_y':'deltay'}, inplace=True)
        pred_labels = pred_labels[pred_labels['confidence'] > conf.validation.confidence_threshold]
        
        height = self.video_info['height']
        true_labels = true_labels[true_labels['ybr'] > conf.validation.disregard_region*height]
        pred_labels = pred_labels[pred_labels['ybr'] > conf.validation.disregard_region*height]
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
        
        return true_labels, pred_labels, motion


def main():
    """TODO: Joao

    """
    vid_characteristics = pd.read_csv(os.path.join(conf.dirs.output, 'online_samples.csv'))
    camera_list = vid_characteristics.camera.unique()
    n_cameras = len(camera_list)
    camera_to_idx = {camera_list[i]:i for i in range(n_cameras)}

    analysis_group_cameras = [Analysis() for _ in range(n_cameras)]

    analysis_group = Analysis()
    
    for filename in os.listdir(conf.dirs.annotations):
        if filename[-4:] != '.csv':
            continue
        vid_ref = filename[:-4]
        logger.info(f'Starting to process the video {vid_ref}.')
        analysis_group.pr_curves_onevid(vid_ref)
        camera = vid_characteristics[vid_characteristics['video_segment_name'] == vid_ref]['camera'].squeeze()
        analysis_group_cameras[camera_to_idx[camera]].pr_curves_onevid(vid_ref)
        logger.info(f'Done processing {vid_ref}.')

    analysis_group.pr_curves_agg()
    for i in range(n_cameras):
        try:
            analysis_group_cameras[i].pr_curves_agg(camera_list[i])
        except ValueError:
            logger.info(f'No objects for camera {camera_list[i]}!')

if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    run_and_catch_exceptions(logger, main)
