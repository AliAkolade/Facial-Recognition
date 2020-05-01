import pickle
import os
import cv2
import dlib
import numpy as np
import onnx
import onnxruntime as ort
from imutils import face_utils
from kivy.clock import Clock, mainthread

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from plyer import filechooser
import time, threading
from pathlib import Path
import traceback
from kivy.properties import StringProperty
import threading



default_directory = os.path.abspath(os.curdir)
default_directory.replace('\\', '/')
# METHODS

def show_log():
    but = Button(text="OK")
    popupWindow = Popup(title="A", title_align='center', content=but, size_hint=(0.5, 0.5))
    popupWindow.open()

def show_popup(xtitle, content, isButton, size):
    popupWindow = Popup(title=xtitle, title_align='center', content=content, size_hint=size)
    if isButton==True:  content.bind(on_press=popupWindow.dismiss)
    #popupWindow.bind(on_dismiss=findFaces)
    #time.sleep(5)
    #popup.dismiss()
    popupWindow.open()


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]
def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]
def findFaces():
    os.chdir(default_directory)
    onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
    fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

    threshold = 0.6

    # load distance
    with open("embeddings/Embeddings.pkl", "rb") as f:
        (saved_embeds, names) = pickle.load(f)
    import tensorflow as tf
    with tf.Graph().as_default():
        with tf.Session() as sess:

            saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
            saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            video_capture = cv2.VideoCapture(0)

            while True:
                #fps = video_capture.get(cv2.CAP_PROP_FPS)
                ret, frame = video_capture.read()

                # preprocess faces
                h, w, _ = frame.shape
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (640, 480))
                img_mean = np.array([127, 127, 127])
                img = (img - img_mean) / 128
                img = np.transpose(img, [2, 0, 1])
                img = np.expand_dims(img, axis=0)
                img = img.astype(np.float32)

                # detect faces
                confidences, boxes = ort_session.run(None, {input_name: img})
                boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                # locate faces
                faces = []
                boxes[boxes < 0] = 0
                for i in range(boxes.shape[0]):
                    box = boxes[i, :]
                    x1, y1, x2, y2 = box

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    aligned_face = fa.align(frame, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                    aligned_face = cv2.resize(aligned_face, (112, 112))

                    aligned_face = aligned_face - 127.5
                    aligned_face = aligned_face * 0.0078125

                    faces.append(aligned_face)

                # face embedding
                if len(faces) > 0:
                    predictions = []

                    faces = np.array(faces)
                    feed_dict = {images_placeholder: faces, phase_train_placeholder: False}
                    embeds = sess.run(embeddings, feed_dict=feed_dict)

                    # prediciton using distance
                    for embedding in embeds:
                        diff = np.subtract(saved_embeds, embedding)
                        dist = np.sum(np.square(diff), 1)
                        idx = np.argmin(dist)
                        if dist[idx] < threshold:
                            predictions.append(names[idx])
                        else:
                            predictions.append("unknown")

                    # draw
                    for i in range(boxes.shape[0]):
                        box = boxes[i, :]

                        text = f"{predictions[i]}"

                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

                cv2.imshow('Live Video Feed', frame)

                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()




class MainWindow(Screen):

    def test(self):
        findFaces()
    def train(self):
        try:
            path = filechooser.open_file(title="Pick a File")[0]
            if path != "":
                self.manager.current = "Training Screen"

        except:
            traceback.print_exc()
            message = Button(text="Please Select Training Files", background_color=(0,0,0,0))
            show_popup("Error", message, True, [0.4, 0.3])
    pass


class TrainWindow(Screen):

    console_output =StringProperty()

    def __init__(self, **kwargs):
        super(TrainWindow, self).__init__(**kwargs)
        self.console_output = "Training Started...\n\n"

    def start_second_thread(self):
        threading.Thread(target=self.setFaces).start()

    @mainthread
    def add_out(self, text):
        self.console_output = str(self.console_output+str(text))

    def setFaces(self):
        os.chdir("..")
        os.chdir("..")
        os.chdir("..")
        TRAINING_BASE = 'faces/training/'
        dirs = os.listdir(TRAINING_BASE)

        os.chdir(default_directory)

        onnx_path = 'models/ultra_light/ultra_light_models/ultra_light_640.onnx'
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        shape_predictor = dlib.shape_predictor('models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
        fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

        images = []
        names = []

        for label in dirs:
            for i, fn in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))):
                self.add_out(str("Loading faces from "+label+"'s data\n"))
                print(f"Start collecting faces from {label}'s data")
                cap = cv2.VideoCapture(os.path.join(TRAINING_BASE, label, fn))
                frame_count = 0
                while True:
                    # read video frame
                    ret, raw_img = cap.read()
                    # process every 5 frames
                    if frame_count % 5 == 0 and raw_img is not None:
                        h, w, _ = raw_img.shape
                        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (640, 480))
                        img_mean = np.array([127, 127, 127])
                        img = (img - img_mean) / 128
                        img = np.transpose(img, [2, 0, 1])
                        img = np.expand_dims(img, axis=0)
                        img = img.astype(np.float32)

                        confidences, boxes = ort_session.run(None, {input_name: img})
                        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                        # if face detected
                        if boxes.shape[0] > 0:
                            x1, y1, x2, y2 = boxes[0, :]
                            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                            aligned_face = fa.align(raw_img, gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                            aligned_face = cv2.resize(aligned_face, (112, 112))

                            cv2.imwrite(f'faces/tmp/{label}_{frame_count}.jpg', aligned_face)

                            aligned_face = aligned_face - 127.5
                            aligned_face = aligned_face * 0.0078125
                            images.append(aligned_face)
                            names.append(label)

                    frame_count += 1
                    if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        break
        import tensorflow as tf
        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.console_output += "\n\nTraining Faces ...\n"
                print("loading checkpoint ...")
                saver = tf.train.import_meta_graph('models/mfn/m1/mfn.ckpt.meta')
                saver.restore(sess, 'models/mfn/m1/mfn.ckpt')

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                embeds = sess.run(embeddings, feed_dict=feed_dict)
                with open("embeddings/embeddings.pkl", "wb") as f:
                    pickle.dump((embeds, names), f)
                self.console_output += "\nDone!\n\n.......................TRAINING COMPLETE......................."
                print("Done!")

    def on_enter(self, *args):
        self.start_second_thread()
    pass


class WindowManager(ScreenManager):
    pass

class Gui(App):
    def build(self):
        return Builder.load_file("gui.kv")


if __name__ == '__main__':
    Gui().run()
