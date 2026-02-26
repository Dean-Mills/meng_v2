# from gat import TinyGAT 
# import torch
# from preprocess import KeypointPreprocessor
# from dataloader import create_coco_dataloader
# from trainer import Trainer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# dataloader_two = create_coco_dataloader(split="train")
# preprocessor = KeypointPreprocessor(device='cuda')
# model = TinyGAT()
# trainer = Trainer(model, preprocessor, device='cuda')
# trainer.train(dataloader_two)
# 

################
# visualizations
################

# from virtual_dataset import VirtualKeypointsDataset
# from settings import settings

# def main():
#     print("="*70)
#     print("VIRTUAL DATASET VISUALIZATION TEST")
#     print("="*70)
    
#     print("\nLoading two_persons dataset...")
#     dataset = VirtualKeypointsDataset(split="two_persons")
    
#     print(f"Dataset size: {len(dataset)}")
#     print(f"Output directory: {settings.output_dir}")
    
#     num_samples = min(5, len(dataset))
#     print(f"\nVisualizing first {num_samples} samples...")
#     print("-"*70)
    
#     for idx in range(num_samples):
#         print(f"\nSample {idx + 1}/{num_samples}:")
        
#         # Get sample
#         sample = dataset[idx]
        
#         print(f"  Image ID: {sample['img_id']}")
#         print(f"  Image shape: {sample['image'].shape}")
#         print(f"  Number of people: {len(sample['keypoints'])}")
#         print(f"  Annotation IDs: {sample['ann_ids']}")
#         print(f"  Distances from camera: {sample['distances']}")
        
#         # Visualize
#         dataset.visualize_item(idx, save_to_file=True)
    
#     print("\n" + "="*70)
#     print("✅ VISUALIZATION COMPLETE!")
#     print(f"Check images in: {settings.output_dir}")
#     print("="*70)

# if __name__ == "__main__":
#     main()
    
    
#main.py
"""Train on virtual dataset"""
from gat import TinyGAT 
import torch
from virtual_preprocess import VirtualKeypointPreprocessor
from virtual_dataloader import create_virtual_dataloader
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataloader = create_virtual_dataloader(split="three_persons", shuffle=True)

preprocessor = VirtualKeypointPreprocessor(device=device)

model = TinyGAT()
trainer = Trainer(model, preprocessor, device=device)

trainer.train(dataloader)