# ============ Base imports ======================
import os
from glob import glob
# ====== External package imports ================
import numpy as np
from scipy import stats
import cv2
# ====== Internal package imports ================
from src.modules.data.database_io import DatabaseIO
#from src.modules.utils.misc
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf = get_config()
# ================================================


def points_between(p1, p2):
    """Generate an list of x/y integer points between p1 and p2

    :param p1: tuple containing (x,y) pair of first point
    :param p2: tuple containing (x,y) pair of second point
    :return: list of tuples of integer points between p1 and p2
    """
    xs = range(p1[0] + 1, p2[0]) or [p1[0]]
    ys = range(p1[1] + 1, p2[1]) or [p1[1]]
    return [(x,y) for x in xs for y in ys]

#input is the sementically segmented image and an nx4 numpy array of x1,y1, x2,y2  coordinates of bounding boxes

def segment_boxes_with_frames(frames, frame_names, boxes):
    """Computes the dominant segment along the bottom edge of each box in boxes

    matches the box name to the locations in frame_names

    :param frames: numpy array (k x 1920 x 1080 x 3) containing image segmentation frames for each of k locations
    :param frame_names: length k list of names, one for each of the k locations
    :param boxes: numpy array where each row is [file name, xtl, ytl, xbr, ybr
    :return: list of segments matching the first index of boxes. segments are "sidewalk", "road", or "other"
    """
    result= []
    
    #boxes[x1,y1,x2,y2]
    for j in np.arange(boxes.shape[0]):
        if j % 1000 == 0:
            logger.info(f"procesing box: {j}")
        # match box name to frame and get frame
        segment = None
        for k, frame_name in enumerate(frame_names):
            if frame_name in boxes[j, 0]:
                segment = k
                break
        if segment is None:
            logger.error(f"Frame not found for box {boxes[j]}")
            result.append("segmentation_frame_not_found")
            continue
        frame = frames[segment]
        # do the thing
        xtl = max(0, int(float(boxes[j, 1])))
        ytl = max(0, int(float(boxes[j, 2])))
        xbr = min(int(float(frame.shape[1]))-1, int(float(boxes[j, 3])))
        ybr = min(int(float(frame.shape[0]))-1, int(float(boxes[j, 4])))
        start= [ybr, xtl]
        end = [ytl, xbr]
        arr=np.array(points_between(start,end))
        arr2 =[]
        for i in np.arange(arr.shape[0]):
            arr2.append(frame[arr[i][0],arr[i][1]])
        arr3=np.array(arr2)
        a=stats.mode(arr3)[0][0][0]
        b=stats.mode(arr3)[0][0][1]
        c=stats.mode(arr3)[0][0][2]

        if a==232 and b==35 and c==244:
            result.append("sidewalk")

        elif a==128 and b==64 and c==128:
             result.append("road")

        else:
            result.append("other")
    return result


def main():
    """Download boxes from the database, run semantic segmentation on them, and upload result to the database

    """
    # load segmented images
    segmented_images_dir = conf.dirs.segmented_images
    segmented_image_names = []
    segmented_images = []
    logger.info("loading semantic segmentations")
    for filename in glob(os.path.join(segmented_images_dir, "*.png")):
        segmented_image_names.append(os.path.basename(filename).split(".")[0])
        segmented_images.append(cv2.imread(filename))
    frames = np.array(segmented_images)  # 7 x 1920 x 1080 x 3

    # get boxes from db
    logger.info("Loading boxes from db")
    dbio = DatabaseIO()
    data, columns = dbio.get_results_boxes()
    ids = [row[-1] for row in data]
    data = [[row[2], row[4], row[5], row[6], row[7]] for row in data]
    data = np.array(data)

    # call segmentation function
    logger.info("segmenting boxes")
    segments = segment_boxes_with_frames(frames, segmented_image_names, data)

    # upload back into database
    dbio.upload_semantic_segments_to_boxes(segments)


if __name__ == "__main__":
    script_name = os.path.basename(__file__).split(".")[0]
    setup(script_name)
    #run_and_catch_exceptions(logger, main)
    main()

