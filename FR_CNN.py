import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import cv2

def fasterrcnn_resnet50_fpn(pretrained=False, progress=True, num_classes=91, **kwargs):
    # Use resnet_fpn_backbone to create the ResNet-50 FPN backbone
    backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained)

    # Create the Faster R-CNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        min_size=800,  # Minimum image size during training
        max_size=1333,  # Maximum image size during training
        image_mean=[0.485, 0.456, 0.406],  # Image mean for normalization
        image_std=[0.229, 0.224, 0.225],  # Image standard deviation for normalization
        **kwargs
    )

    if pretrained:
        # Load pre-trained weights using torch.hub
        state_dict = torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth', progress=progress)
        model.load_state_dict(state_dict)

    return model

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Initialize the video capture object
video_path = 'A:\mojor_project_sem8\nyolonas_deepsort\data\video\test.mp4'  # Update with your video file path
cap = cv2.VideoCapture(video_path)

# Check if video capture is successful
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Define the codec and create VideoWriter object
output_path = 'A:\mojor_project_sem8\nyolonas_deepsort\output\out3.mp4'
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to a PyTorch tensor
    img_tensor = torchvision.transforms.functional.to_tensor(frame)
    
    # Perform inference
    with torch.no_grad():
        predictions = model([img_tensor])
    
    # Draw bounding boxes on the frame
    for pred in predictions[0]['boxes']:
        box = pred.detach().numpy().astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label = 'Object'  # Replace with the actual class name if available
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Write the frame into the output video
    out.write(frame)

# Release video capture object and release the output video
cap.release()
out.release()
cv2.destroyAllWindows()
