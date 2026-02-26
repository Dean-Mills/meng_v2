def test_dataloader(dataloader):
    num_batches = len(dataloader)
    print(f"Number of batches in DataLoader: {num_batches}")

    print("\nTesting DataLoader with batch size: 4")
    for batch in dataloader:
        images = batch['image']
        keypoints_list = batch['keypoints']
        img_ids = batch['img_id']
        
        print(f"Number of images in batch: {len(images)}")
        
        for i, img in enumerate(images):
            print(f"  Image {i} shape: {img.shape}")
        
        for i, kps in enumerate(keypoints_list):
            print(f"  Image {i} (ID: {img_ids[i]}) has {len(kps)} people")
            
            for p, person_kps in enumerate(kps):
                visible_kps = (person_kps[:, 2] > 0).sum().item()
                print(f"    Person {p} has {visible_kps}/17 visible keypoints")
        
        break

def test_depth_estimator(dataloader):
    from depth_estimation import estimate_depth, visualize_depth

    for batch in dataloader:
        image = batch['image'][0]
        img_id = batch['img_id'][0]
        
        print(f"Processing image ID: {img_id}")
        
        depth_map = estimate_depth(image)
        
        from settings import settings 
        output_path = f"{settings.output_dir}/depth_estimation_{img_id}.jpg"
        
        visualize_depth(image, depth_map, output_path)
        print(f"Depth visualization saved to: {output_path}")
        
        break

def test_keypoint_rcnn(dataloader):
    """Test the KeypointRCNN model on images from the dataloader"""
    from pose_rcnn import detect_keypoints, visualize_keypoints
    
    for batch in dataloader:
        images = batch['image']
        img_ids = batch['img_id']
        
        print(f"Processing {len(images)} images")
        
        for i, img in enumerate(images):
            img_id = img_ids[i]
            print(f"  Processing image {i} (ID: {img_id})")
            
            boxes, keypoints, scores = detect_keypoints(img)
            
            print(f"    Detected {len(boxes)} people with confidence > 0.7")
            
            visualize_keypoints(img, boxes, keypoints, img_id)
            
        break