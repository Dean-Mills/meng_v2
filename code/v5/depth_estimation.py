import torch
import matplotlib.pyplot as plt

def estimate_depth(depth_model, image, depth_transform, device):
    """
    Estimate depth map using a pre-loaded model.
    
    Args:
        depth_model: The pre-loaded MiDaS model.
        image: RGB image tensor [C, H, W] to be processed.
        depth_transform: The torchvision transform required by the MiDaS model.
        device: The device ('cuda' or 'cpu') to run inference on.
    
    Returns:
        depth_map: Depth map tensor [H, W] on the CPU.
    """
    image_numpy = image.permute(1, 2, 0).cpu().numpy()

    transformed_tensor = depth_transform(image_numpy)
    
    input_batch = transformed_tensor.to(device)

    with torch.no_grad():
        prediction = depth_model(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[1:], # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu()
    
    return depth_map

def visualize_depth(image, depth_map, output_path=None):
    """
    Visualize RGB image and corresponding depth map. This function does not need changes.
    """
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    img_np = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')
    
    # Depth map
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map.numpy(), cmap='plasma')
    plt.title("Depth Map")
    plt.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    try:
        plt.show()
    except Exception:
        pass
    
    plt.close()

# import torch
# import matplotlib.pyplot as plt

# def estimate_depth(image):
#     """
#     Estimate depth map from a single RGB image using a pre-trained model
    
#     Args:
#         image: RGB image tensor [C, H, W]
    
#     Returns:
#         depth_map: Depth map tensor [H, W]
#     """

#     image = image.float() / 255.0

#     model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
#     model.eval()
    
#     if torch.cuda.is_available():
#         model.cuda()
#         image = image.cuda()
    
#     from torchvision import transforms
    
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                              std=[0.229, 0.224, 0.225])
#     ])
    
#     input_batch = transform(image).unsqueeze(0)
    
#     with torch.no_grad():
#         prediction = model(input_batch)
        
#         if prediction.shape[2:] != image.shape[1:]:
#             prediction = torch.nn.functional.interpolate(
#                 prediction.unsqueeze(1),
#                 size=image.shape[1:],
#                 mode="bicubic",
#                 align_corners=False,
#             ).squeeze(1)
    
#     depth_map = prediction.squeeze().cpu()
    
#     return depth_map

# def visualize_depth(image, depth_map, output_path=None):
#     """
#     Visualize RGB image and corresponding depth map
#     """
#     plt.figure(figsize=(12, 5))
    
#     # Original image
#     plt.subplot(1, 2, 1)
#     img_np = image.permute(1, 2, 0).cpu().numpy()
#     plt.imshow(img_np)
#     plt.title("Original Image")
#     plt.axis('off')
    
#     # Depth map
#     plt.subplot(1, 2, 2)
#     plt.imshow(depth_map.numpy(), cmap='plasma')
#     plt.title("Depth Map")
#     plt.axis('off')
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path)
    
#     try:
#         plt.show()
#     except:
#         pass
    
#     plt.close()
