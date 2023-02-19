import cv2
import keyboard
import time


def take_picture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Show to screen
        cv2.imshow('OpenCV Feed', frame)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # Save Picture
        if keyboard.is_pressed("space"):
            timestr = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(timestr+"_"+"SavedPicture.png", frame)

    cap.release()
    cv2.destroyAllWindows()


# take_picture()
