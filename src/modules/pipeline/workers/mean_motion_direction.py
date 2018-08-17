# ============ Base imports ======================
# ====== External package imports ================
import cv2
import pandas as pd
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


class MeanMotionDirection(PipelineWorker):
    """Takes the output from "lk_sparse_optical_flow" and "yolo3_detect" to associate xy points and displacement vectors from the first with boxes from the second. It then calculates the mean 
    motion of an object in a bounding box by averaging the angles and magnitudes of the displacement vectors.
    """
    
    def initialize(self, annotate_result_frame_key, stationary_threshold, *args, **kwargs):
        """Initialize the worker with the results from the annotated frame that contains box and optical flow information
        """
        self.annotate_result_frame_key = annotate_result_frame_key
        self.stationary_threshold = stationary_threshold

    def startup(self):
        """Startup
        """
        pass

    def run(self, item):
        """Implement the task. This code retrieves the tracked points and boxes from lk sparse optical flow and object 
        detection/classification, loops through the boxes and associates the points with their appropriate boxes. It then takes the mean 
        position and direction for each points and returns these values.
        """
        # Get the tracked points from the frame dictionary
        tracked_points = item['tracked_points']
        boxes = item['boxes']
        columns = ['mean_x', 'mean_y', 'mean_delta_x', 'mean_delta_y', 'angle_from_vertical', 'magnitude']
        item['points_grouped_by_box_header'] = columns

        # The output from the previous worker's optical flow calculations returns a list of arrays that contain the xy coordinates of a point, 
        # and the x, y changes representing a vector pointing to that point from the previous frame. This for loop loops through each array,
        # converts the array to a dataframe, and renames the columns. It then runs another for loop (detailed below) that appends box IDs to the rows
        if tracked_points is None:
            self.logger.info(f"{item['video_info']['file_name']}, frame {item['frame_number']}: No tracked points found")
            item['points_grouped_by_box'] = None
        else:
            points_grouped_by_box = np.zeros((boxes.shape[0], 6))
            # Loop through each box
            for j in range(len(boxes)):
                # Initialize empty lists for x and y points. These will be used to store points that match boxes
                points_x = []
                points_y = []
                points_dx = []
                points_dy = []
                # Loop through all of the rows in an array
                for k in range(len(tracked_points)):
                    # Check if the points are within the box points. Be sure to match box indices to their proper corners.
                    if float(boxes[j, 0]) < float(tracked_points[k][0]) < float(boxes[j, 2]) and \
                            float(boxes[j, 1]) < float(tracked_points[k][1]) < float(boxes[j, 3]):
                        # If there is a match, append the matching delta_x and delta_y points to the empty lists
                        points_x.append(tracked_points[k][0])
                        points_y.append(tracked_points[k][1])
                        points_dx.append(tracked_points[k][2])
                        points_dy.append(tracked_points[k][3])
                # Average the xy and delta x, delta y values
                x_ave = np.mean(points_x) if len(points_x) > 0 else 0
                y_ave = np.mean(points_y) if len(points_y) > 0 else 0
                dx_ave = np.mean(points_dx) if len(points_dx) > 0 else 0
                dy_ave = np.mean(points_dy) if len(points_dy) > 0 else 0
                angle_from_vertical = np.arctan2(-dy_ave, dx_ave)
                magnitude = np.sqrt(dx_ave**2 + dy_ave**2)
                # if motion is below threshold set to zero
                magnitude = magnitude if magnitude < self.stationary_threshold else 0

                # add to array
                points_grouped_by_box[j] = [x_ave, y_ave, dx_ave, dy_ave, angle_from_vertical, magnitude]

            # Return the list
            item['points_grouped_by_box'] = points_grouped_by_box
        self.done_with_item(item)

    # Define shutdown
    def shutdown(self):
        """Shutdown
        """
        pass

