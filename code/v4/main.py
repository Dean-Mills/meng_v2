from gat import TinyGAT 
import torch
from preprocess import KeypointPreprocessor
from dataloader import create_coco_dataloader
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataloader_two = create_coco_dataloader(split="train")
preprocessor = KeypointPreprocessor(device='cuda')
model = TinyGAT()
trainer = Trainer(model, preprocessor, device='cuda')
trainer.train(dataloader_two)