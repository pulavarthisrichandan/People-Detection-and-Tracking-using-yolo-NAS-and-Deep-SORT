import numpy as np
import datetime
import cv2
import torch
from absl import app, flags
from deep_sort_realtime.deepsort_tracker import DeepSort
from super_gradients.training import models
from super_gradients.common.object_names import Models
from collections import OrderedDict

# Define command line flags
flags.DEFINE_string('model', 'yolo_nas_l', 'yolo_nas_l or yolo_nas_m or yolo_nas_s')
flags.DEFINE_float('conf', 0.50, 'confidence threshold')

FLAGS = flags.FLAGS

def main(_argv):
    # Initialize the webcam
    video_cap = cv2.VideoCapture(0)  # 0 for default webcam, change if using an external camera

    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=20, nn_budget=30, max_iou_distance=0.7)

    # Check if GPU is available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the YOLO model
    model = models.get(FLAGS.model, pretrained_weights="coco").to(device)

    # Load the COCO class labels the YOLO model was trained on
    classes_path = "./configs/coco.names"
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")

    while True:
        # Start time to compute the FPS
        start = datetime.datetime.now()
        
        # Read a frame from the webcam
        ret, frame = video_cap.read()

        # If there is no frame, break the loop
        if not ret:
            print("No frame captured...")
            break

        # Run the YOLO model on the frame
        detect = model.predict(frame, iou=0.5, conf=FLAGS.conf)

        # Extract detection results
        bboxes_xyxy = torch.from_numpy(detect.prediction.bboxes_xyxy).tolist()
        confidence = torch.from_numpy(detect.prediction.confidence).tolist()
        labels = torch.from_numpy(detect.prediction.labels).tolist()
        concate = [sublist + [element] for sublist, element in zip(bboxes_xyxy, confidence)]
        final_prediction = [sublist + [element] for sublist, element in zip(concate, labels)]

        # Initialize the list of bounding boxes and confidences
        results = []

        # Loop over the detections
        for data in final_prediction:
            confidence = data[4]
            if float(confidence) < FLAGS.conf:
                continue
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

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
            x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, str(track_id) + " - " + str(class_names[class_id]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # End time to compute the FPS
        end = datetime.datetime.now()
        
        # Calculate and display FPS
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 255, 0), 2)

        # Show the frame
        cv2.imshow("Webcam Object Tracking", frame)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release webcam object
    video_cap.release()

    # Close all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
