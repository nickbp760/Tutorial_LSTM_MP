import numpy as np
import cv2
import mediapipe as mp
from keyPointMP import mediapipe_detection
# from keyPointMP import draw_styled_landmarks_face
from Face_Distraction.getDataTraining_Face import extract_keypoints_face
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh

colors = [(245, 17, 16), (117, 245, 16), (16, 117, 245), (100, 50, 16), (15, 17, 16), (245, 200, 200)]
actions = np.array(['LirikKanan', 'LirikKiri', 'MenolehKanan', 'MenolehKiri', 'Normal', 'TutupMata'])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


def model_reload():
    model = Sequential()
    model.add(GRU(256, return_sequences=True, activation='tanh', input_shape=(50, 15)))
    model.add(Dropout(0.2))
    model.add(GRU(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(GRU(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.load_weights('Weight_model/actionGRUNikoCheatDataReal_ReportGRU2.h5')
    return model


def real_time_camera_predict():
    # 1. New detection variables
    sequence = []
    predictions = []
    sentence = ""
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    model = model_reload()
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, face_mesh)
            # print(results)
            # Draw landmarks
            # draw_styled_landmarks_face(image, results)
            # 2. Prediction logic
            keypoints = extract_keypoints_face(results, image)
        #         sequence.insert(0,keypoints, image)
        #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-50:]

            if len(sequence) == 50:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        sentence = actions[np.argmax(res)]
                        print(sentence)

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, sentence, (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


real_time_camera_predict()
