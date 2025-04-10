from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import time
import csv
import os
from collections import deque
from threading import Thread
from utils.signal_processing import bandpass_filter, calculate_bpm
from utils.face_detection import detect_face_and_roi
from utils.visualization import start_live_plot
from scipy import signal
import cv2
import numpy as np
# import mediapipe as mp
# # from ultralytics import YOLO

mixer.init()

if not os.path.exists("music.wav"):
    raise FileNotFoundError("Missing 'music.wav'. Place it in the same directory as this script!")

mixer.music.load("music.wav")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

THRESH = 0.25
FRAME_CHECK = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise Exception("Cannot open webcam. Check camera permissions or try changing the index.")

flag = 0


# cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
BUFFER_SIZE = int(FPS * 10)
BANDPASS_FREQ = (0.7, 3.0)

green_intensity_buffer = deque(maxlen=BUFFER_SIZE)
live_data = {
    "time": [],
    "bpm": [],
    "face_angle": [],
    "green_intensity": []
}

os.makedirs('data', exist_ok=True)
timestamp = time.strftime("%Y%m%d-%H%M%S")
LOG_FILE = f'data/heart_rate_log_{timestamp}.csv'
with open(LOG_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "HeartRate", "FaceYaw"])

last_bpm = None
last_bpm_time = 0
start_time = time.time()

plot_thread = Thread(target=start_live_plot, args=(live_data,), daemon=True)
plot_thread.start()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detect(gray)

    for face in faces:
        shape = predict(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if ear < THRESH:
            flag += 1
            print(f"[WARNING] Drowsiness frame count: {flag}")

            if flag >= FRAME_CHECK:
                cv2.putText(frame, "*WARNING!**WARNING!**WARNING!**WARNING!**WARNING!*", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Alert! eye lid close detected! Possibility of driver fatigue", (8, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "or drowsiness or Distraction (Phone)", (8, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, "*WARNING!**WARNING!**WARNING!**WARNING!**WARNING!*", (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not mixer.music.get_busy():
                    mixer.music.play()

        else:
            flag = 0

    cv2.imshow("Drowsiness Detection", frame)

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time() - start_time

    forehead_roi, left_cheek_roi, right_cheek_roi, yaw, face_bbox = detect_face_and_roi(frame)

    if face_bbox is not None:
        x, y, w_box, h_box = face_bbox
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

    if forehead_roi is not None:
        avg_green = np.mean(forehead_roi[:, :, 1])
        green_intensity_buffer.append(avg_green)
        live_data["green_intensity"].append(avg_green)

    if len(green_intensity_buffer) == BUFFER_SIZE:
        detrended_signal = signal.detrend(green_intensity_buffer)
        filtered_signal = bandpass_filter(detrended_signal, FPS, *BANDPASS_FREQ)
        bpm = calculate_bpm(filtered_signal, FPS)
        if bpm:
            last_bpm = int(bpm)
            last_bpm_time = time.time()

            with open(LOG_FILE, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), last_bpm, yaw])

    if last_bpm and (time.time() - last_bpm_time < 3):
        live_data["time"].append(current_time)
        live_data["bpm"].append(last_bpm)
        live_data["face_angle"].append(yaw)

    info_text = f"BPM: {last_bpm if last_bpm else 'No detection'} | Yaw: {yaw:.1f}°"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    direction = "Face: Left" if yaw < -5 else "Face: Right" if yaw > 5 else "Face: Center"
    cv2.putText(frame, direction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Tracking", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plot_thread.join()
