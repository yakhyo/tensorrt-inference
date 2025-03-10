from models import RetinaFace

# Initialize the RetinaFace model
uniface_inference = RetinaFace(
    model_path="weights/retinaface_mv2_static.onnx",    # Model path
    conf_thresh=0.5,                        # Confidence threshold
    pre_nms_topk=5000,                      # Pre-NMS Top-K detections
    nms_thresh=0.4,                         # NMS IoU threshold
    post_nms_topk=750,                      # Post-NMS Top-K detections
    dynamic_size=False,                     # Arbitrary image size inference
    input_size=(640, 640)                   # Pre-defined input image size
)


import cv2
from uniface.visualization import draw_detections

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Perform inference
    boxes, landmarks = uniface_inference.detect(frame)

    # Draw detections on the frame
    draw_detections(frame, (boxes, landmarks), vis_threshold=0.6)

    # Display the output
    cv2.imshow("Webcam Inference", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()