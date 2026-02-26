from dataset import CocoKeypointsDataset
from dataloader import create_coco_dataloader
from preprocess import KeypointPreprocessor
from trainer import Trainer
from gat import DepthAwareGAT
import torch

def main():
    """Lean main training script"""
    
    print("Starting COCO Keypoint Training...")
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load data
    dataloader = create_coco_dataloader(split="val", batch_size=4, num_workers=0)
    print(f"Data loaded: {len(dataloader)} batches")
    
    # Preprocess
    preprocessor = KeypointPreprocessor(device=device)
    print("Preprocessor ready")
    
    # Create GAT
    model = DepthAwareGAT(input_dim=5, hidden_dim=32, output_dim=16)
    print(f"GAT created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train
    trainer = Trainer(model=model, preprocessor=preprocessor, device=device)
    trainer.train(dataloader=dataloader, epochs=50, lr=0.001)
    
    print("Training complete!")

def test_preprocessor():
    """Test our simplified preprocessor"""
    
    # Step 1: Create dataloader (small batch for testing)
    print("Creating dataloader...")
    dataloader = create_coco_dataloader(
        split="val", 
        batch_size=2,  # Small batch for testing
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues during testing
    )
    
    # Step 2: Create preprocessor
    print("Creating preprocessor...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    preprocessor = KeypointPreprocessor(device=device)
    
    # Step 3: Get one batch and test preprocessing
    print("Getting first batch...")
    batch = next(iter(dataloader))
    
    print(f"Batch info:")
    print(f"  Images shape: {batch['image'].shape}")
    print(f"  Number of images: {len(batch['keypoints'])}")
    print(f"  People per image: {[len(img_kps) for img_kps in batch['keypoints']]}")
    
    # Step 4: Process the batch
    print("\nProcessing batch...")
    try:
        pyg_graphs = preprocessor.process_batch(batch)
        
        print(f"Success! Created {len(pyg_graphs)} graphs")
        
        # Examine first graph
        if pyg_graphs:
            graph = pyg_graphs[1]
            print(f"\nFirst graph info:")
            print(f"  Nodes: {graph.x.shape[0]}")
            print(f"  Node features shape: {graph.x.shape}")
            print(f"  People in graph: {graph.num_people}")
            print(f"  Person labels: {graph.person_labels}")
            print(f"  Sample node features:\n{graph.x[:3]}")  # First 3 nodes
            
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()