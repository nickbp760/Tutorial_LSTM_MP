import cv2
import time


def take_video(frameNumLimit: int):
    cap = cv2.VideoCapture(0)
    # We need to check if camera
    # is opened previously or not
    if (cap.isOpened() is False):
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    timestr = time.strftime("%Y%m%d-%H%M%S")
    result = cv2.VideoWriter(timestr+"_"+"SavedVideo.avi", cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

    # frameNum for calculation
    frameNum = 0

    while cap.isOpened() and frameNum <= frameNumLimit:

        # Read feed
        ret, frame = cap.read()

        if ret:
            # Write the frame into the
            # file 'filename.avi'
            result.write(frame)

            # Display the frame
            # saved in the file
            cv2.imshow('Frame', frame)
            frameNum = frameNum + 1

            # Press q on keyboard
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture and video
    # write objects
    cap.release()
    result.release()
    cv2.destroyAllWindows()


# 10 fps in default current webcam, please calculate this in this below link
# if I set the 300 fps limit then it wolud be 30s
# https://webcamtests.com/fps
# take_video(300)
