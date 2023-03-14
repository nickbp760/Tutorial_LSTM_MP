import os
import cv2
import mediapipe as mp
import numpy as np
from keyPointMP import mediapipe_detection
from Face_Distraction.getDataTraining_Face import extract_keypoints_face


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def take_keypoints_Completeface_from_video(DATA_PATH: str, action: str, sequnce: int, videoFilePath: str = None):
    # create a VideoCapture object to read the video file
    cap = cv2.VideoCapture(videoFilePath)

    # check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    frame_num = 0
    # Set mediapipe model
    while cap.isOpened():
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Read feed
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # NEW Export keypoints
                keypoints = extract_keypoints_face(results, image)
                # print(keypoints.shape)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                frame_num += 1
            else:
                break

    # Release the video file and close the window
    cap.release()
    cv2.destroyAllWindows()


# specify the directory path
path = "./FaceVideoDataset"
DATA_PATH = os.path.join('Complete_Face_Data')
# use os.listdir() to get a list of all files and folders in the directory
filesAndFolders = os.listdir(path)

# use a loop to iterate over the list and print out only the folders
for item in filesAndFolders:
    print(item)
    sequence = 0
    fullPath = os.path.join(path, item)  # get the full path of the item
    # if os.path.isdir(fullPath):  # check if the item is a directory
    #     print(item)
    Personfolders = os.listdir(fullPath)
    for personVideo in Personfolders:
        print("     ", personVideo)
        fullPathVideoList = os.path.join(fullPath, personVideo)  # get the full path of the item
        VideoList = os.listdir(fullPathVideoList)
        for videoFileName in VideoList:
            print("             ", videoFileName)
            fullPathVideoFileName = os.path.join(fullPathVideoList, videoFileName)  # get the full path of the item
            os.makedirs(os.path.join(DATA_PATH, item, str(sequence)))
            take_keypoints_Completeface_from_video(DATA_PATH, item, sequence, fullPathVideoFileName)
            sequence += 1
