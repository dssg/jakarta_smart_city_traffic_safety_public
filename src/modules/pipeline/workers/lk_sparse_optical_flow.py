# ============ Base imports ======================
# ====== External package imports ================
import numpy as np
import cv2
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


class LKSparseOpticalFlow(PipelineWorker):
    """Detects corners on objects in a frame using Shi-Tomasi method and computes optical flow from the previous frame
    using Lucas-Kanade algorithm
    
    Returns an object that contains arrays for each detected point containing their x and y positions, and x and y
    vectors pointing to those positions
    
    """
    def initialize(self, frame_key, annotate_frame_key, annotate_result_frame_key, new_point_detect_interval, path_track_length,
                   good_flow_difference_threshold, new_point_occlusion_radius, bg_mask_key, winSize, maxLevel, maxCorners,
                   qualityLevel, minDistance, blockSize, backward_pass, new_point_detect_interval_per_second,
                   how_many_track_new_points_before_clearing_points, **kwargs):
        """Initialize the worker with information from the frame and parameters set by user.
        :frame_key: The frame that will be analyzed
        :annotate_frame_key: If the frame had previous operations applied to it, these will be passed these into the optical flow method
        :annotate_result_frame_key: The result of optical flow will be stored here
        :new_point_detect_interval: How often (in terms of number of frames) Shi-Tomasi should be used to detect points in a frame.
        :path_track_length: How many previous points to store while calculating and displaying flow. By default, 10
        :good_flow_difference_threshold: Absolute difference dhreshold to determine "good" points to track after applying Shi-Tomasi. By default, 1
        :new_point_occlusion_radius: Radius around detected points, smaller radius will result in more points and vice versa. By default, 5
        :bg_mask_key: Optional parameter to pass the result of a background subtraction (Mixture of Gaussian, K Nearest Neighbor etc.). Might help with corner detection.
        :winSize: size of the search window at each pyramid level.
        :maxLevel: 	0-based maximal pyramid level number; if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; if pyramids are passed to input then
        algorithm will use as many levels as pyramids have but no more than maxLevel.
        :maxCorners: The maximum number of corners to track
        :qualityLevel: Minimum corner quality below which corner will be rejected
        :minDistance: Throws away corners within the minimum distance of a found corner
        :blockSize: Neighborhood size around point
        :backward_pass: Logical value as to whether to do a "backward pass" optical flow. By default, the process calculates optical flow from frame 1 to frame 2, and reverse flow from frame 2 to
        frame 1. This improve accuracy, but increases computation time.
        :new_point_detect_interval_per_second: How often (in seconds) Shi-Tomasi should be used to detect points in a frame
        :how_many_track_new_points_before_clearing_points: How many points should be retained until the process clears them. By default, 10
        :**kwargs: Additional arguments
        """
        self.frame_key = frame_key
        self.annotate_frame_key = annotate_frame_key
        self.annotate_result_frame_key = annotate_result_frame_key
        self.track_len = path_track_length
        self.bg_mask_key = bg_mask_key
        self.backward_pass = backward_pass
        if new_point_detect_interval is not None and new_point_detect_interval_per_second is not None:
            self.logger.error("Cannot specify both new_point_detect_interval and new_point_detect_interval_per_second")
        self.new_point_detect_interval = new_point_detect_interval
        self.new_point_detect_interval_per_second = new_point_detect_interval_per_second
        self.good_flow_difference_threshold = good_flow_difference_threshold
        self.how_many_track_new_points_before_clearing_points = how_many_track_new_points_before_clearing_points
        self.new_point_occlusion_radius = new_point_occlusion_radius
        #TODO: how to encode cv2 options in config file?
        self.lk_params = dict( winSize  = tuple(winSize), maxLevel = maxLevel,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict( maxCorners = maxCorners,
                               qualityLevel = qualityLevel,
                               minDistance = minDistance,
                               blockSize = blockSize)

    def startup(self):
        """Define worker startup parameters.
        """
        self.prev_gray = None
        self.frame_idx = 1
        self.tracks = []
        self.fps = []
        self.vid_info = None
        self.track_new_points_count = 0

    def run(self, item):
        """Have the worker implement the task. The steps are:
        1. Read a frame from the frame dictionary and apply grayscale + other annotation
        2. Detect corners and choose the best ones
        3. Calculate optical flow from frame 1 to frame 2 (and possibly vice versa)
        4. Store results and optionally draw optical flow polylines on video
        """
        # Get frame from frame_dict
        frame = item[self.frame_key]
        # if this is a new video, start detection right away and update fps
        if (self.vid_info is None) or (self.vid_info["file_name"] != item["video_info"]["file_name"]):
            self.vid_info = item["video_info"]
            self.fps = self.vid_info["fps"]
            self.frame_idx = 1
            self.track_new_points_count = 0
        points = None

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a copy of the original frame
        if self.annotate_frame_key in item.keys():
            #self.logger.info("using previous annotated_frame")
            vis = item[self.annotate_frame_key]
        else:
            #self.logger.info("creating new annotated")
            vis = frame.copy()

        if self.prev_gray is None:
            self.logger.info(f"{item['video_info']['file_name']}, frame {item['frame_number']}: No previous frame, skipping motion detection")
            item["tracked_points"] = None
        else:
            # Check if there is a background subtracted mask
            if self.bg_mask_key is not None:
                bg_mask = item[self.bg_mask_key]
                #self.logger.info("using background mask")
            else:
                bg_mask = None
                #self.logger.info("not using background mask")


            # If statement if tracks has elements in it
            if len(self.tracks) > 0:

                # Create a previous image and a next image
                img0, img1 = self.prev_gray, frame_gray

                # For tracks in tracks, reshape to 1 by 2 (for coordinate pairs)
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)

                #self.logger.info("calculate forward Optical flow")
                # Calculate optical flow from img0 to img1
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, bg_mask, **self.lk_params)

                if self.backward_pass:
                    #self.logger.info("calculate backward Optical flow")
                    # Calculate optical flow from img1 to img0
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, bg_mask, **self.lk_params)

                    # Get the absolute value of the coordinates * the optical flow
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)

                    # Define a "good" optical flow as a difference less than 1
                    good = d < self.good_flow_difference_threshold
                    #self.logger.info(f"{sum(good)} out of {len(good)} points are good")

                    # Calculate displacement
                    p0a = p0.reshape(-1, 2)[good]
                    p1a = p1.reshape(-1, 2)[good]
                else:
                    # Calculate displacement
                    p0a = p0.reshape(-1, 2)
                    p1a = p1.reshape(-1, 2)
                    good = [True] * p0a.shape[0]

                displacement = p1a - p0a
                points = np.concatenate([p1a, displacement], axis = 1)

                # Create an empty object for new tracks
                new_tracks = []

                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    # Append the x, y coordinates
                    tr.append((x, y))

                    # If the tracks are greater than 10, delete them
                    if len(tr) > self.track_len:
                        del tr[0]

                    # Append the tracks to the empty "new tracks"
                    new_tracks.append(tr)

                    # Draw a circle around the point
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

                # Overwrite tracks with new tracks
                self.tracks = new_tracks

                # Draw a line between two points
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 0, 0))

        fraction_of_second = (self.frame_idx / self.vid_info["fps"]) % 1
        #self.logger.info(f"{item['video_info']['file_name']}, frame {item['frame_number']}: frac:{fraction_of_second}, frac mod interval: {fraction_of_second % (1/self.new_point_detect_interval_per_second)}, 1/fps:{1/self.vid_info['fps']}")
        if ((self.new_point_detect_interval is not None) and
            (self.frame_idx % self.new_point_detect_interval == 0)) \
            or \
            ((self.new_point_detect_interval_per_second is not None) and
             (fraction_of_second % (1/self.new_point_detect_interval_per_second) <= 1/self.vid_info["fps"])):
            self.logger.debug(f"{item['video_info']['file_name']}, frame {item['frame_number']}: finding new points")
            self.track_new_points_count += 1
            if self.track_new_points_count == self.how_many_track_new_points_before_clearing_points:
                self.tracks = []
                self.track_new_points_count = 0
                self.logger.debug(f"{item['video_info']['file_name']}, frame {item['frame_number']}: clearing points")
            # Create an empty mask with the same dimensions as the frame
            mask = np.zeros_like(frame_gray)
            mask[:] = 255

            # Draw circles with the mask
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), self.new_point_occlusion_radius, 0, -1)

            # Find good features to track
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    self.tracks.append([(x, y)])
            self.logger.debug(f"{len(self.tracks)} points found")

        item[self.annotate_result_frame_key] = vis
        item['tracked_points'] = points

        self.prev_gray = frame_gray
        self.frame_idx += 1
        self.done_with_item(item)

    def shutdown(self):
        """Shutdown the worker when task is complete.
        """
        pass
