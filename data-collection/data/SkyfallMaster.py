file_string = '''

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
import json

detection_graph = tf.Graph()
sys.path.append("..")


# Load a frozen infrerence graph into memory
def load_inference_graph(NUM_CLASSES, PATH_TO_CKPT, PATH_TO_LABELS):
    # load label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # load frozen tensorflow model into memory
    print("> ====== loading frozen graph into memory", PATH_TO_CKPT)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Hand Inference graph loaded.", PATH_TO_CKPT, PATH_TO_LABELS)
    return detection_graph, sess, category_index


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def get_tags(classes, category_index, num_hands_detect, score_thresh, scores, boxes, image_np):
    im_height, im_width, channels = image_np.shape
    tags = []
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'N/A'

            tag = {}
            tag['class'] = class_name
            tag['score'] = scores[i].tolist()
            tag['box'] = boxes[i].tolist()
            tag["box_center"] = ( int((left + (right - left)/2 )) , int((top + (bottom - top)/2 )) )
            tags.append(tag)

    return tags

# Actual detection .. generate scores and bounding boxes given an image


def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)

# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, image_np):
    im_height, im_width, channels = image_np.shape
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

def draw_box_on_image_id(tags, image_np): 
    for tag in tags:
        cv2.putText(image_np, str(tag["id_label"]) , tag["box_center"],
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
        cv2.putText(image_np,  "  " + str(tag["class"]), tag["box_center"],
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 9), 2)

# Show fps value on image.
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


 


# Code to thread reading camera input.
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    # Worker threads that process video frame
    def worker(input_q, output_q, cap_params, frame_processed):
        print(">> loading frozen model for worker")
        
        detection_graph, sess, category_index = detector_utils.load_inference_graph(num_classes, frozen_graph_path, label_path)
        sess = tf.Session(graph=detection_graph)
        while True:
            #print("> ===== in worker loop, frame ", frame_processed)
            frame = input_q.get()
            if (frame is not None):
                # actual detection
                boxes, scores, classes = detector_utils.detect_objects(
                    frame, detection_graph, sess)
                
                tags = detector_utils.get_tags(classes, category_index, num_hands_detect, score_thresh, scores, boxes, frame)
                
                if (len(tags) > 0):
                    id_utils.get_id(tags, seen_object_list)
                    web_socket_client.send_message(tags,"hand")

                id_utils.refresh_seen_object_list(seen_object_list, object_refresh_timeout)
                detector_utils.draw_box_on_image_id(tags, frame) 
                
                output_q.put(frame)
                frame_processed += 1
            else:
                output_q.put(frame)
        sess.close()
    
    def load_labelmap(path):
        """Loads label map proto.

        Args:
        path: path to StringIntLabelMap proto text file.
        Returns:
        a StringIntLabelMapProto
        """
        with tf.gfile.GFile(path, 'r') as fid:
            label_map_string = fid.read()
            label_map = string_int_label_map_pb2.StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)
        _validate_label_map(label_map)
        return label_map

'''