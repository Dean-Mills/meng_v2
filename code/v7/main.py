#main.py
"""Train on virtual dataset"""
from gat import TinyGAT 
import torch
from virtual_preprocess import VirtualKeypointPreprocessor
from virtual_dataloader import create_virtual_dataloader
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataloader = create_virtual_dataloader(split="four_persons", shuffle=True)

preprocessor = VirtualKeypointPreprocessor(device=device)

model = TinyGAT()
trainer = Trainer(model, preprocessor, device=device)

trainer.train(dataloader)