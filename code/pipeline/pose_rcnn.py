import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from settings import settings

def detect_keypoints(image, confidence_threshold=0.7):
    """
    Detect human keypoints in an image using KeypointRCNN
    
    Args:
        image: Image tensor [C, H, W]
        confidence_threshold: Minimum confidence score for detections
        
    Returns:
        boxes: Bounding boxes [N, 4]
        keypoints: Keypoints [N, 17, 3] (x, y, score)
        scores: Detection confidence scores [N]
    """
    # Load model (lazy loading - will only happen once)
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # Convert image to float
    img_float = image.float() / 255.0
    
    # Make prediction
    with torch.no_grad():
        prediction = model([img_float])[0]
    
    # Extract predictions
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    keypoints = prediction['keypoints'].cpu().numpy()
    
    # Filter predictions by confidence score
    confident_detections = scores > confidence_threshold
    
    boxes = boxes[confident_detections]
    scores = scores[confident_detections]
    keypoints = keypoints[confident_detections]
    
    return boxes, keypoints, scores

def visualize_keypoints(image, boxes, keypoints, img_id=None):
    """
    Visualize keypoint predictions
    
    Args:
        image: Image tensor [C, H, W]
        boxes: Bounding boxes [N, 4]
        keypoints: Keypoints [N, 17, 3]
        img_id: Optional image ID for saving
        
    Returns:
        output_path: Path where the visualization was saved (if img_id provided)
    """
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Display image
    img_np = image.permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    
    if img_id:
        plt.title(f"KeypointRCNN Detections - Image ID: {img_id}")
    else:
        plt.title("KeypointRCNN Detections")
    
    # Colors for different people
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
    
    # Draw bounding boxes and keypoints
    for j, (box, kps) in enumerate(zip(boxes, keypoints)):
        color = colors[j % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], c=color, linewidth=2)
        
        # Draw keypoints
        # For each keypoint, the format is [x, y, score]
        for k, (x, y, v) in enumerate(kps):
            if v > 0.5:  # Only show confident keypoints
                plt.scatter(x, y, c=color, s=20, marker='o')
        
        # Draw skeleton connections
        connections = [
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            (5, 7), (7, 9),  # left shoulder to left elbow to left wrist
            (6, 8), (8, 10),  # right shoulder to right elbow to right wrist
            (5, 6), (5, 11), (6, 12),  # shoulders to hips
            (11, 13), (13, 15),  # left hip to left knee to left ankle
            (12, 14), (14, 16)  # right hip to right knee to right ankle
        ]
        
        for connection in connections:
            # Check if both keypoints are confident
            if kps[connection[0], 2] > 0.5 and kps[connection[1], 2] > 0.5:
                plt.plot(
                    [kps[connection[0], 0], kps[connection[1], 0]],
                    [kps[connection[0], 1], kps[connection[1], 1]],
                    c=color, linewidth=1
                )
    
    # Save visualization if image ID is provided
    if img_id:
        output_path = f"{settings.output_dir}/keypoint_rcnn_{img_id}.jpg"
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    
    try:
        plt.show()
    except:
        pass
    
    plt.close()
    
    if img_id:
        return output_path