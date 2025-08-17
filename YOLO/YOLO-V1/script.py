import cv2
from utils import model, prepare_input, get_detection, obj_to_idx
from model import YOLOV1
# define camera
rtsp_cam = 'rtsp://admin:admin123@192.168.1.108/live'
cam = cv2.VideoCapture(rtsp_cam )

# Define desired width and height for the custom shape
desired_width = 448
desired_height = 448


while True:
    # getting the current frame 
    check, frame = cam.read()

    if not check:
        print("Failed to grab frame. Check your RTSP link.")
        break

    # resized frame with defined size
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    print(f"Shape of the captured frame: {resized_frame.shape}")
    f_tensor = prepare_input(resized_frame, 448)
    print(f"Shape after the conversion: {f_tensor.shape}")

    # display the frame 
    cv2.imshow('video', resized_frame) 

    # pass the frame into model to get object detection 
    det = get_detection(model, resized_frame, f_tensor, conf_threshold=0.2, nms_threshold=0.5, obj_to_idx=obj_to_idx, score_threshold=0.5)



    ########
    # we can apply different types of transformation to the captured frames
    ########

    combined_frame = cv2.hconcat([resized_frame, det])

    cv2.imshow('multiple frame' ,combined_frame)
    # cv2.imshow('detection', det)

    key = cv2.waitKey(1)
    # press Esc to break 
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()