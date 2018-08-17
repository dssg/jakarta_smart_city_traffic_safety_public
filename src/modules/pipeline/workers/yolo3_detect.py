# ============ Base imports ======================
import os
import struct
# ====== External package imports ================
import numpy as np
import colorsys
import cv2
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import tensorflow as tf
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


class Yolo3Detect(PipelineWorker):
    """Implementation of YOLOv3 for object detection and classification.
    """
    def initialize(self, buffer_size, gpu, gpu_share, weights_path, frame_key, annotate_result_frame_key,
                   object_detect_threshold, class_nonzero_threshold, non_maximal_box_suppression,
                   non_maximal_box_suppression_threshold, annotation_font_scale, **kwargs):
        """
        font_scale = 
        buffer_size = how many frames to wait for
        frame_key = frame ID
        annotate_result_frame_key = result from previous process, if applicable
        weights_path = path to weights file from pre-trained model
        gpu = activate GPU
        gpu_frac = how much of the GPU should be used in %
        object_detect_threshold = confidence threshold for drawing a bounding box
        class_nonzero_threshold = confidence threshold for keeping bounding boxes
        net_h
        net_w
        anchors = initial boxes to anchor objects (?)
        labels = labels from pre-trained model
        important_labels = labels we expect to use for traffic detection/classification
        """
        self.font_scale = annotation_font_scale
        self.buffer_size = buffer_size
        self.frame_key = frame_key
        self.annotate_result_frame_key = annotate_result_frame_key
        self.weights_path = weights_path
        self.gpu = gpu
        self.gpu_frac = gpu_share
        self.object_detect_threshold = object_detect_threshold
        self.class_nonzero_threshold = class_nonzero_threshold
        self.non_maximal_box_suppression = non_maximal_box_suppression
        self.non_maximal_box_suppression_threshold = non_maximal_box_suppression_threshold
        # other things needed for this Yolo model
        self.net_h, self.net_w = 416, 416
        self.anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        self.labels = ["pedestrian", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                       "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                       "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                       "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                       "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                       "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                       "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                       "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        self.important_labels = ["pedestrian", "bicycle", "car", "motorbike", "bus", "train", "truck"]
        self.important_label_indices = np.array([lab in self.important_labels for lab in self.labels])
        # set up colors for annotating frames
        n = len(self.important_labels)
        hsv_tuples = [(x*1.0/n, 0.5, 0.5) for x in range(n)]
        self.label_colors = list(map(lambda x: tuple([255*val for val in colorsys.hsv_to_rgb(*x)]), hsv_tuples))


    def startup(self):
        """Startup. Set up keras tensorflow backend to utilize cuda.
        """
        # must import in this process
        from keras.backend.tensorflow_backend import set_session
        # set environment variables
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.gpu)
        # specify fraction of gpu memory to use
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_frac
        set_session(tf.Session(config=config))
        # create model and load weights
        self.yolov3 = Yolo3Detect.make_yolov3_model()
        if self.weights_path is not None:
            self.logger.info("Loading YOLO Weights")
            weight_reader = Yolo3Detect.WeightReader(self.weights_path)
            weight_reader.load_weights(self.yolov3)
        # create buffers
        self.buffer_fill = 0
        self.buffer = []
        self.failed_draw = 0
        self.box_id = 0

    def run(self, item):
        """Draw boxes around objects (depending on confidence threshold) and assign class predictions based on most confident prediction.
        """
        self.buffer.append(item)
        self.buffer_fill += 1
        if self.buffer_fill == self.buffer_size:
            self.logger.info("Buffer full, processing")
            frames = np.concatenate([np.expand_dims(item[self.frame_key], axis=0) for item in self.buffer], axis=0)
            # detect on this batch of images
            annotated_frames, boxeses, objectnesseses, predicted_classeses, classify_scoreses = self.detect_images(frames, self.buffer)
            # attach detection info to frame dictionaries
            for i, item in enumerate(self.buffer):
                item[self.annotate_result_frame_key] = annotated_frames[i]
                item["boxes"] = boxeses[i]
                num_boxes = len(item["boxes"])
                box_ids = range(self.box_id, self.box_id + num_boxes)
                self.box_id += num_boxes
                item["box_id"] = box_ids
                item["object_classes"] = self.important_labels
                item["classes"] = self.important_labels
                item["objectness"] = objectnesseses[i]
                item["predicted_classes"] = predicted_classeses[i]
                item["classify_scores"] = classify_scoreses[i]
                item["boxes_header"] = ["xtl", "ytl", "xbr", "ybr", "objectness"] + self.important_labels
                self.done_with_item(item)
            self.buffer_fill = 0
            self.buffer = []

    def shutdown(self):
        """Wait for the buffer to fill, then shut down the process.
        """
        if self.buffer_fill != 0:
            frames = np.concatenate([np.expand_dims(item[self.frame_key], axis=0) for item in self.buffer], axis=0)
            self.logger.debug("cleaning out buffer: frames shape:{}".format(frames.shape))
            annotated_frames, boxeses, objectnesseses, predicted_classeses, classify_scoreses = self.detect_images(frames, self.buffer)
            for i, item in enumerate(self.buffer):
                item[self.annotate_result_frame_key] = annotated_frames[i]
                item["boxes"] = boxeses[i]
                num_boxes = len(item["boxes"])
                box_ids = range(self.box_id, self.box_id + num_boxes)
                self.box_id += num_boxes
                item["box_id"] = box_ids
                item["object_classes"] = self.important_labels
                item["classes"] = self.important_labels
                item["objectness"] = objectnesseses[i]
                item["predicted_classes"] = predicted_classeses[i]
                item["classify_scores"] = classify_scoreses[i]
                item["boxes_header"] = ["xtl", "ytl", "xbr", "ybr", "objectness"] + self.important_labels
                self.done_with_item(item)
            self.buffer_fill = 0
            self.buffer = []
        self.logger.error(f"Could not draw {self.failed_draw} boxes")

    #==============================
    #= Support Functions/Classes ==
    #==============================
    def detect_images(self, images, items):
        """Detect objects in an image and draws boxes
        """
        n_images, image_h, image_w, _ = images.shape
        # yolos is a len 3 list with predictions at 3 different scales (see paper)
        preproc = Yolo3Detect.preprocess_inputs(images, self.net_h, self.net_w)
        self.logger.debug("Starting batch detect")
        yolos = self.yolov3.predict(preproc, batch_size=self.buffer_size)
        self.logger.debug("Finished batch detect")
        all_boxes = []
        # decode the output of the network at all three scales
        for i in range(len(yolos)):
            all_boxes.append(Yolo3Detect.decode_netout(yolos[i], self.anchors[i], self.important_label_indices,
                                                         self.net_h, self.net_w))
        # combine bounding boxes from 3 different scales
        all_boxes = np.concatenate(all_boxes, 1)
        # split up into individual images
        images_boxes = np.split(all_boxes, np.arange(1, n_images))
        images = np.split(images, np.arange(1, n_images))
        images_boxes_out = []
        images_out = []
        images_objectness = []
        images_predicted_classes = []
        images_classify_scores = []
        # for the boxes from each image...
        for image_boxes, image, item in zip(images_boxes, images, items):
            # TODO: this breaks when there is just one image:
            image_boxes = np.squeeze(image_boxes)
            # TODO: this breaks when there is just one image:
            image = np.squeeze(image)
            # keep boxes with an object prob above detection threshold and at least one class prob above the class
            # threshold
            keep_bc_object = image_boxes[:, 4] > self.object_detect_threshold
            keep_bc_classes = np.any(image_boxes[:, 5:] > self.class_nonzero_threshold, axis=1)
            keep_boxes = keep_bc_object * keep_bc_classes
            image_boxes = image_boxes[keep_boxes, :]
            if image_boxes.shape[0] != 0:
                # scale bounding boxes to image size
                image_boxes = Yolo3Detect.correct_yolo_boxes(image_boxes, image_h, image_w, self.net_h, self.net_w)
                # suppress non-maximal boxes (with IOU over threshold)
                if self.non_maximal_box_suppression:
                    image_boxes = Yolo3Detect.do_nms_boxes(image_boxes, self.non_maximal_box_suppression_threshold)
                    # redo class nonzero thresholding
                    keep_bc_classes = np.any(image_boxes[:, 5:] > self.class_nonzero_threshold, axis=1)
                    image_boxes = image_boxes[keep_bc_classes, :]
                # draw bounding boxes on the image using labels
                image, failed = Yolo3Detect.draw_boxes(image, image_boxes, self.important_labels, self.label_colors, (0, 255, 0), self.font_scale, self.class_nonzero_threshold, self.logger)
                self.failed_draw += failed
                if failed != 0:
                    self.logger.error(f"Video {item['video_info']['file_name']}, frame {item['frame_number']} failed to draw {failed} boxes")
            images_out.append(image)
            images_boxes_out.append(image_boxes)
            if image_boxes.shape[0] >0:
                images_objectness.append(image_boxes[:,4])
                images_predicted_classes.append([self.important_labels[amax] for amax in np.argmax(image_boxes[:, 5:], 1)])
                images_classify_scores.append(np.max(image_boxes[:, 5:], 1))
            else:
                images_objectness.append(np.array([[]]))
                images_predicted_classes.append(np.array([[]]))
                images_classify_scores.append(np.array([[]]))
        self.logger.debug("done detecting: len(images_out):{}, len(image_boxes_out):{}".format(len(images_out), len(images_boxes_out)))
        return images, images_boxes_out, images_objectness, images_predicted_classes, images_classify_scores

    class WeightReader:
        """Reads pre-trained weights for neural network.
        """
        def __init__(self, weight_file):
            with open(weight_file, 'rb') as w_f:
                major, = struct.unpack('i', w_f.read(4))
                minor, = struct.unpack('i', w_f.read(4))
                revision, = struct.unpack('i', w_f.read(4))

                if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                    w_f.read(8)
                else:
                    w_f.read(4)

                transpose = (major > 1000) or (minor > 1000)

                binary = w_f.read()

            self.offset = 0
            self.all_weights = np.frombuffer(binary, dtype='float32')

        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset - size:self.offset]

        def load_weights(self, model):
            for i in range(106):
                try:
                    conv_layer = model.get_layer('conv_' + str(i))

                    if i not in [81, 93, 105]:
                        norm_layer = model.get_layer('bnorm_' + str(i))

                        size = np.prod(norm_layer.get_weights()[0].shape)

                        beta = self.read_bytes(size)  # bias
                        gamma = self.read_bytes(size)  # scale
                        mean = self.read_bytes(size)  # mean
                        var = self.read_bytes(size)  # variance

                        weights = norm_layer.set_weights([gamma, beta, mean, var])

                    if len(conv_layer.get_weights()) > 1:
                        bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                        kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                        kernel = kernel.transpose([2, 3, 1, 0])
                        conv_layer.set_weights([kernel, bias])
                    else:
                        kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                        kernel = kernel.transpose([2, 3, 1, 0])
                        conv_layer.set_weights([kernel])
                except ValueError:
                    pass

        def reset(self):
            self.offset = 0

    @staticmethod
    def _conv_block(inp, convs, skip=True):
        x = inp
        count = 0

        for conv in convs:
            if count == (len(convs) - 2) and skip:
                skip_connection = x
            count += 1

            if conv['stride'] > 1: x = ZeroPadding2D(((1, 0), (1, 0)))(
                x)  # peculiar padding as darknet prefer left and top
            x = Conv2D(conv['filter'],
                       conv['kernel'],
                       strides=conv['stride'],
                       padding='valid' if conv['stride'] > 1 else 'same',
                       # peculiar padding as darknet prefer left and top
                       name='conv_' + str(conv['layer_idx']),
                       use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

        return add([skip_connection, x]) if skip else x

    @staticmethod
    def _interval_overlap(interval_a, interval_b):
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

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def bbox_iou(box1, box2):
        intersect_w = Yolo3Detect._interval_overlap([box1[0], box1[2]], [box2[0], box2[2]])
        intersect_h = Yolo3Detect._interval_overlap([box1[1], box1[3]], [box2[1], box2[3]])

        intersect = intersect_w * intersect_h

        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    @staticmethod
    def make_yolov3_model():
        """Make cnn model (?)
        """
        input_image = Input(shape=(None, None, 3))

        # Layer  0 => 4
        x = Yolo3Detect._conv_block(input_image,
                                      [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                       {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                       {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                       {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

        # Layer  5 => 8
        x = Yolo3Detect._conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                                          {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                                          {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

        # Layer  9 => 11
        x = Yolo3Detect._conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                                          {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

        # Layer 12 => 15
        x = Yolo3Detect._conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                                          {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                                          {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

        # Layer 16 => 36
        for i in range(7):
            x = Yolo3Detect._conv_block(x, [
                {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])

        skip_36 = x

        # Layer 37 => 40
        x = Yolo3Detect._conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                                          {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                                          {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

        # Layer 41 => 61
        for i in range(7):
            x = Yolo3Detect._conv_block(x, [
                {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])

        skip_61 = x

        # Layer 62 => 65
        x = Yolo3Detect._conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                                          {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                                          {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

        # Layer 66 => 74
        for i in range(3):
            x = Yolo3Detect._conv_block(x, [
                {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])

        # Layer 75 => 79
        x = Yolo3Detect._conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                                          {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                                          {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                                          {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                                          {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                                      skip=False)

        # Layer 80 => 82
        yolo_82 = Yolo3Detect._conv_block(x, [
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
            {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

        # Layer 83 => 86
        x = Yolo3Detect._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                                      skip=False)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_61])

        # Layer 87 => 91
        x = Yolo3Detect._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                                          {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                                          {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                                          {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                                          {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                                      skip=False)

        # Layer 92 => 94
        yolo_94 = Yolo3Detect._conv_block(x,
                                            [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92},
                                             {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                                              'layer_idx': 93}], skip=False)

        # Layer 95 => 98
        x = Yolo3Detect._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                                      skip=False)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_36])

        # Layer 99 => 106
        yolo_106 = Yolo3Detect._conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 100},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 101},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 102},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 103},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 104},
            {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

        model = Model(input_image, [yolo_82, yolo_94, yolo_106])
        return model

    @staticmethod
    def preprocess_inputs(images, net_h, net_w):
        new_h, new_w, _ = images[0].shape

        # determine the new size of the image
        if (float(net_w) / new_w) < (float(net_h) / new_h):
            new_h = (new_h * net_w) / new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h) / new_h
            new_h = net_h

        # resize the images to the new size
        new_images = np.empty((images.shape[0], net_h, net_w, 3))
        for i in range(images.shape[0]):
            resized = cv2.resize(images[i, :, :, ::-1] / 255., (int(new_w), int(new_h)))

            # embed the image into the standard letter box
            new_images[i, int((net_h - new_h) // 2):int((net_h + new_h) // 2),
            int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized

        return new_images

    @staticmethod
    def decode_netout(netout, anchors, important_label_indices, net_h, net_w):
        n_images, grid_h, grid_w, = netout.shape[:3]
        cell_scale = np.array([[grid_w, grid_h]])
        net_dims = np.array([[net_w, net_h]])
        anchors = np.reshape(np.array(anchors), (1, 3, 2))
        # 3 bounding boxes per grid cell
        nb_box = 3
        netout = netout.reshape((n_images, grid_h, grid_w, nb_box, -1))

        # select out only important classes
        netout = netout[:, :, :, :, np.hstack([[True] * 5, important_label_indices])]
        # sigmoid the x and y positions of bounding boxes
        netout[..., :2] = Yolo3Detect._sigmoid(netout[..., :2])
        # sigmoid the objectness and class probs
        netout[..., 4:] = Yolo3Detect._sigmoid(netout[..., 4:])
        # classes
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]

        images_boxes = []
        # loop through each grid cell
        for g in range(grid_h * grid_w):
            row = g / grid_w
            col = g % grid_w
            cell_pos = np.array([[[col, row]]])

            # 5th element is objectness score
            objectness = netout[:, int(row), int(col), :, 4]
            classes = netout[:, int(row), int(col), :, 5:]

            # box dimensions are x, y, w, and h
            box_dims = netout[:, int(row), int(col), :, :4]
            # x, y, w, h = netout[:][int(row)][int(col)][:][:4]

            box_dims[:, :, :2] = (box_dims[:, :, :2] + cell_pos) / cell_scale
            box_dims[:, :, 2:] = anchors * np.exp(box_dims[:, :, 2:]) / net_dims
            # x = (col + x) / grid_w  # center position, unit: image width
            # y = (row + y) / grid_h  # center position, unit: image height
            # w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            # h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # last elements are class probabilities
            #classes = netout[:, int(row), col, b, 5:, important_label_indices

            # switch representation to top-left / bottom-right from center/heigh-width
            cell_boxes = np.concatenate([box_dims[:, :, :2] - box_dims[:, :, 2:]/2, box_dims[:, :, :2] + box_dims[:, :, 2:]/2, np.expand_dims(objectness, -1), classes], -1)

            images_boxes.append(cell_boxes)
        #            boxes.append(np.concatenate([box_dims[:, ]]
        #                [np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness]), + classes]))
        images_boxes = np.concatenate(images_boxes, 1)
        # remove boxes with zero score in all categories (doesn't work with multiple images
        return images_boxes

    @staticmethod
    def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        if (float(net_w) / image_w) < (float(net_h) / image_h):
            new_w = net_w
            new_h = (image_h * net_w) / image_w
        else:
            new_h = net_w
            new_w = (image_w * net_h) / image_h
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        scale = np.array([[image_w / x_scale, image_h / y_scale, image_w / x_scale, image_h / y_scale]])
        shift = np.array([[x_offset, y_offset, x_offset, y_offset]])

        boxes[:, :4] = (boxes[:, :4]-shift) * scale

        return boxes

    @staticmethod
    def do_nms_boxes(boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = boxes.shape[1] - 5
        else:
            return
        # for each class
        for c in range(nb_class):
            # sort from most to least probability boxes
            sorted_indices = np.argsort(-boxes[:, 5+c])

            # for most to least likely boxes
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                # if class probability zero, ignore
                if boxes[index_i, 5+c] == 0: continue

                # for all lesser probability boxes of that class
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if boxes[index_j, 5+c] == 0: continue

                    # if intersection over union is above threshold, zero the prob on the lower prob box
                    if Yolo3Detect.bbox_iou(boxes[index_i, :4], boxes[index_j, :4]) >= nms_thresh:
                        boxes[index_j, 5+c] = 0
        return boxes

    @staticmethod
    def draw_boxes(image, boxes, labels, class_colors, box_color=None, font_scale=0.5, draw_threshold=0, logger=logger):
        failed = 0
        np.set_printoptions(precision=2)
        k = len(labels)
        if box_color is None:
            box_color = (0, 255, 0)
        # for each box
        for row in range(boxes.shape[0]):
            try:
                box = boxes[row, :]
                # show bounding box and obj score regardless of class probabilities
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, 2)
                cv2.putText(image, "{:0.2f}".format(box[4]), (int(box[0]), int(box[3])), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            font_scale, box_color, 1)
                # add labels for nonzero classes
                j = 0
                for i, label in enumerate(labels):
                    # if nonzero class prob then print it
                    if box[5+i] > draw_threshold:
                        j += 1
                        label_str = "{}:{:0.2f}".format(label, box[5+i])
                        #cv2.putText(image, "{}".format(label_str), (int(box[0]), int(box[3]) + int(font_scale*j*25)),
                        #            cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, class_colors[i], 1)
                        cv2.putText(image, "{}".format(label_str), (int(box[0]), int(box[3]) + int(font_scale*j*25)),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, box_color, 1)
            except Exception as e:
                logger.error(f"Could not draw rectangle: ({box[0]}, {box[1]}), ({box[2]}, ({box[3]})")
                failed += 1
        return image, failed
