from models import RetinaFaceTRT
import cv2
from utils.visualization import draw_detections

# Initialize the RetinaFace TensorRT model
uniface_inference = RetinaFaceTRT(
    engine_path="weights/model.engine",   # TensorRT engine path
    conf_thresh=0.5,                                   # Confidence threshold
    pre_nms_topk=5000,                                 # Pre-NMS Top-K detections
    nms_thresh=0.4,                                    # NMS IoU threshold
    post_nms_topk=750,                                 # Post-NMS Top-K detections
    dynamic_size=False,                                # Fixed image size inference
    input_size=(640, 640)                              # Pre-defined input size
)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Perform face detection using TensorRT inference
    boxes, landmarks = uniface_inference.detect(frame)


    # Draw detections
    draw_detections(frame, (boxes, landmarks), vis_threshold=0.6)

    # Display frame
    cv2.imshow("TensorRT Webcam Inference", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
