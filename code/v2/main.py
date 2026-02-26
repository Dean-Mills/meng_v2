from gat import DepthAwareGAT
from preprocess import KeypointPreprocessor
from dataloader import create_coco_dataloader
from trainer import Trainer

dataloader_two = create_coco_dataloader(split="train")
preprocessor = KeypointPreprocessor(device='cuda')
model = DepthAwareGAT()
trainer = Trainer(model, preprocessor, device='cuda')
trainer.train(dataloader_two)