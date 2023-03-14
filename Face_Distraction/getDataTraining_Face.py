import numpy as np
import os
import cv2
import mediapipe as mp
from keyPointMP import mediapipe_detection, draw_styled_landmarks_face
from Normalisation import normalisation_faceLandmark


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('Face_Data')

# Actions that we try to detect
actions = np.array(['HadapKanan', 'HadapKiri', 'HadapDepan'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


def create_folder():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except Exception:
                pass


# create_folder()


def extract_keypoints_face(results, image):
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark])
        # print(face.shape)
        face = normalisation_faceLandmark(face, image)
        face = face.flatten()
        # print(face.shape)
    else:
        face = np.zeros(468*3)
    return np.concatenate([face])


def take_keypoints_face_from_video(DATA_PATH: str, videoFilePath: str = None):
    if videoFilePath is None:
        cap = cv2.VideoCapture(0)
    else:
        # create a VideoCapture object to read the video file
        cap = cv2.VideoCapture(videoFilePath)

    # check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks_face(image, results)

                    # NEW Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)

                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence),
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)

                    # NEW Export keypoints
                    keypoints = extract_keypoints_face(results, image)
                    # print(keypoints.shape)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break


# take_keypoints_face_from_video(DATA_PATH)


def load_image_face_detection():
    image = cv2.imread("20230224-100951_SavedPicture.png")
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # Make detections
        image, results = mediapipe_detection(image, holistic)
        # print(results)

        # Draw landmarks
        draw_styled_landmarks_face(image, results)
        extract_keypoints_face(results, image)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)
        # To hold the window on screen, we use cv2.waitKey method
        # Once it detected the close input, it will release the control
        # To the next line
        # First Parameter is for holding screen for specified milliseconds
        # It should be positive integer. If 0 pass an parameter, then it will
        # hold the screen until user close it.
        cv2.waitKey(0)
        # It is for removing/deleting created GUI window from screen
        # and memory
        cv2.destroyAllWindows()


# load_image_face_detection()
