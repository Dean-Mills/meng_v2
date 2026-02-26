from pose_rcnn import detect_keypoints, visualize_keypoints
from depth_estimation import estimate_depth, visualize_depth
from settings import settings
import numpy as np

def process_single_image(image, img_id):
    """
    Process a single image through both depth estimation and keypoint detection
    
    Args:
        image: Image tensor [C, H, W]
        img_id: Image ID for saving outputs
        
    Returns:
        depth_map: Estimated depth map
        keypoints: Detected keypoints [N, 17, 3]
        scores: Detection confidence scores [N]
    """
    print(f"Processing image ID: {img_id}")
    
    # 1. Estimate depth
    depth_map = estimate_depth(image)
    # depth_path = f"{settings.output_dir}/depth_{img_id}.jpg"
    # visualize_depth(image, depth_map, depth_path)
    # print(f"  Depth map saved to: {depth_path}")
    
    # 2. Detect keypoints
    boxes, keypoints, scores = detect_keypoints(image)
    # keypoints_path = f"{settings.output_dir}/keypoints_{img_id}.jpg"
    # visualize_keypoints(image, boxes, keypoints, img_id)
    # print(f"  Keypoints saved to: {keypoints_path}")
    
    return depth_map, keypoints, scores

def build_graph_features(depth_map, keypoints):
    """
    Build graph features for GAT from depth map and keypoints
    
    Args:
        depth_map: Depth map tensor [H, W]
        keypoints: Keypoints [N, 17, 3] (x, y, score)
        
    Returns:
        node_features: Features for each keypoint (node) in the graph
    """
    node_features = []
    
    for person_kps in keypoints:
        person_features = []
        
        for kp in person_kps:
            x, y, score = kp
            
            if score < 0.5:
                kp_features = np.zeros(4)
            else:
                x_int, y_int = int(round(x)), int(round(y))
                
                height, width = depth_map.shape
                x_int = max(0, min(x_int, width - 1))
                y_int = max(0, min(y_int, height - 1))
                
                depth_val = depth_map[y_int, x_int].item()
                
                kp_features = np.array([x, y, depth_val, score])
            
            person_features.append(kp_features)
        
        node_features.append(np.stack(person_features))
    
    return node_features

def process_batch(dataloader, num_batches=1):
    """Process multiple batches of images"""
    all_node_features = []
    batch_count = 0
    
    for batch in dataloader:
        images = batch['image']
        img_ids = batch['img_id']
        
        print(f"Processing batch {batch_count+1}/{num_batches}")
        
        for i, (img, img_id) in enumerate(zip(images, img_ids)):
            print(f"  Processing image {i+1}/{len(images)} (ID: {img_id})")
            
            depth_map = estimate_depth(img)
            
            boxes, keypoints, scores = detect_keypoints(img)
            
            for person_idx, person_kps in enumerate(keypoints):
                person_features = []
                
                for kp_idx, (x, y, score) in enumerate(person_kps):
                    if score > 0.5:  # Only consider confident keypoints
                        x_int, y_int = int(round(x)), int(round(y))
                        
                        h, w = depth_map.shape
                        x_int = max(0, min(x_int, w - 1))
                        y_int = max(0, min(y_int, h - 1))
                        
                        depth_val = depth_map[y_int, x_int].item()
                        
                        feature = np.array([x, y, depth_val, score])
                    else:
                        # For low-confidence keypoints, use zeros
                        feature = np.zeros(4)
                    
                    person_features.append(feature)
                
                all_node_features.append(np.array(person_features))
            
            print(f"Found {len(keypoints)} people in image {img_id}")
        
        batch_count += 1
        if batch_count >= num_batches:
            break
    
    return all_node_features