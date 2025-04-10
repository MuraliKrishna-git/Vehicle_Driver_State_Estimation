import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,  # Lower for greater sensitivity
    min_tracking_confidence=0.2
)


def detect_face_and_roi(frame):
    """
    Processes the frame to detect the face and extract regions of interest.
    Returns:
      - forehead_roi: The cropped image region of the forehead.
      - left_cheek_roi: The cropped image region of the left cheek.
      - right_cheek_roi: The cropped image region of the right cheek.
      - yaw: The computed yaw angle (in degrees) of the face.
      - face_bbox: The overall face bounding box as (x, y, width, height).
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Compute face bounding box from all landmarks
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x = int(min(xs) * w)
        max_x = int(max(xs) * w)
        min_y = int(min(ys) * h)
        max_y = int(max(ys) * h)
        face_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Try to extract forehead ROI using key landmarks
        try:
            forehead_center = landmarks[10]  # approximate center of forehead
            left_bound = landmarks[332]  # approximate left edge landmark
            right_bound = landmarks[103]  # approximate right edge landmark
            vertical_ref = landmarks[9]  # vertical reference (between forehead and eyes)
        except IndexError:
            return None, None, None, 0.0, face_bbox

        forehead_x, forehead_y = int(forehead_center.x * w), int(forehead_center.y * h)
        # Estimate forehead dimensions; adjust scaling as needed
        forehead_width = int((right_bound.x - left_bound.x) * w * 1.2)
        forehead_height = int((forehead_center.y - vertical_ref.y) * h * 1.5)

        top = max(0, forehead_y - forehead_height // 2)
        bottom = min(h, forehead_y + forehead_height // 2)
        left_coord = max(0, forehead_x - forehead_width // 2)
        right_coord = min(w, forehead_x + forehead_width // 2)

        # If computed ROI is invalid, fallback to upper portion of face bbox
        if top >= bottom or left_coord >= right_coord:
            fallback_bottom = min_y + int(0.3 * (max_y - min_y))
            forehead_roi = frame[min_y:fallback_bottom, min_x:max_x]
        else:
            forehead_roi = frame[top:bottom, left_coord:right_coord]

        # Extract cheek ROIs using landmarks (if available)
        try:
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
        except IndexError:
            left_cheek = right_cheek = None

        cheek_size = 20
        if left_cheek is not None:
            left_cheek_x = int(left_cheek.x * w)
            left_cheek_y = int(left_cheek.y * h)
            left_cheek_roi = frame[
                             max(0, left_cheek_y - cheek_size):min(h, left_cheek_y + cheek_size),
                             max(0, left_cheek_x - cheek_size):min(w, left_cheek_x + cheek_size)
                             ]
        else:
            left_cheek_roi = None

        if right_cheek is not None:
            right_cheek_x = int(right_cheek.x * w)
            right_cheek_y = int(right_cheek.y * h)
            right_cheek_roi = frame[
                              max(0, right_cheek_y - cheek_size):min(h, right_cheek_y + cheek_size),
                              max(0, right_cheek_x - cheek_size):min(w, right_cheek_x + cheek_size)
                              ]
        else:
            right_cheek_roi = None

        # Compute yaw angle using the eyes and nose landmarks
        try:
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]
            eye_center_x = ((left_eye.x + right_eye.x) / 2) * w
            nose_x = nose_tip.x * w
            yaw = np.degrees(np.arctan2(nose_x - eye_center_x, w * 0.1))
        except IndexError:
            yaw = 0.0

        return forehead_roi, left_cheek_roi, right_cheek_roi, yaw, face_bbox

    else:
        return None, None, None, 0.0, None


def release():
    face_mesh.close()
