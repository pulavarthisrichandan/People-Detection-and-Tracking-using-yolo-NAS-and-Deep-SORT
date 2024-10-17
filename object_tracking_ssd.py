from absl import app, flags
import torch
import torchvision
from torchvision import transforms as tf
from PIL import Image
import cv2
import numpy as np
import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import OrderedDict
from absl import flags
import sys

# Define command line flags
flags.DEFINE_float('conf', 0.5, 'confidence threshold')

# Parse the command line flags
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# Initialize the webcam
video_cap = cv2.VideoCapture(0)  # 0 for default webcam, change if using an external camera

# Initialize the DeepSort tracker
tracker = DeepSort(max_age=20, nn_budget=30, max_iou_distance=0.7)

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
model = torchvision.models.detection.ssd300_vgg16(pretrained = True)
model.eval().to(device)

# Load the COCO class labels the Faster R-CNN model was trained on
coco_names = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", 
              "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
              "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
              "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", 
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
              "toothbrush"]

while True:
    # Start time to compute the FPS
    start = datetime.datetime.now()
    
    # Read a frame from the webcam
    ret, frame = video_cap.read()

    # If there is no frame, break the loop
    if not ret:
        print("No frame captured...")
        break

    # Convert frame to tensor and move to device
    frame_tensor = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Run the Faster R-CNN model on the frame
    with torch.no_grad():
        detect = model(frame_tensor)

    # Extract detection results
    boxes = detect[0]["boxes"].cpu().numpy()
    labels = detect[0]["labels"].cpu().numpy()
    scores = detect[0]["scores"].cpu().numpy()

    # Initialize the list of bounding boxes and confidences
    results = []

    # Loop over the detections
    for box, label, score in zip(boxes, labels, scores):
        if score < FLAGS.conf:
            continue
        xmin, ymin, xmax, ymax = box.astype(int)
        class_id = label
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], score, class_id])

    # Update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)
    
    # Initialize dictionary to store merged tracks
    merged_tracks = OrderedDict()

    # Loop over the tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        
        # Check if class_id is valid
        if class_id < len(coco_names):
            class_name = coco_names[class_id]
        else:
            class_name = "Unknown"
        
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(track_id) + " - " + class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Merge tracks based on proximity
        if len(merged_tracks) == 0:
            merged_tracks[track_id] = [x1, y1, x2, y2]
        else:
            merged = False
            for existing_id, bbox in merged_tracks.items():
                if abs((x1 + x2) // 2 - (bbox[0] + bbox[2]) // 2) < 50 and abs((y1 + y2) // 2 - (bbox[1] + bbox[3]) // 2) < 50:
                    merged_tracks[existing_id] = [(x1 + bbox[0]) // 2, (y1 + bbox[1]) // 2, (x2 + bbox[2]) // 2, (y2 + bbox[3]) // 2]
                    merged = True
                    break
            if not merged:
                merged_tracks[track_id] = [x1, y1, x2, y2]

    # Draw merged tracks
    for track_id, bbox in merged_tracks.items():
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # End time to compute the FPS
    end = datetime.datetime.now()
    
    # Calculate and display FPS
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Webcam ObjectTracking", frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam object
video_cap.release()

# Close all windows
cv2.destroyAllWindows()


