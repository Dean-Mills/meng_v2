# import matplotlib
# matplotlib.use('TkAgg')
# from dataloader import create_coco_dataloader
# from test import test_dataloader, test_depth_estimator, test_keypoint_rcnn
# from utils import process_single_image, build_graph_features, process_batch
# import numpy as np
# from gat import prepare_dataset, train_gat
# from settings import settings
# import torch

# def main():
#     dataloader = create_coco_dataloader(
#         split='train', 
#         batch_size=4
#     )
    
#     # test_dataloader(dataloader)
#     # test_depth_estimator(dataloader)
#     # test_keypoint_rcnn(dataloader)
    
#     print("Extracting features from images...")
#     all_node_features = process_batch(dataloader, num_batches=20)
#     print(f"Extracted features for {len(all_node_features)} people")
    
#     if len(all_node_features) > 0:
#         print("\nPreparing dataset for GAT training...")
#         dataset = prepare_dataset(all_node_features)
        
#         print("\nTraining GAT model...")
#         model = train_gat(dataset, epochs=300)
        
#         model_path = f"{settings.output_dir}/pose_gat_model.pt"
#         torch.save(model.state_dict(), model_path)
#         print(f"Model saved to {model_path}")
#     else:
#         print("No valid features extracted. Cannot train GAT model.")
    
# if __name__ == "__main__":
#     main()
    
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import numpy as np
# from tqdm import tqdm

# from dataloader import create_coco_dataloader
# from settings import settings
# from basic.preprocess import KeypointPreprocessor
# from basic.gat import DepthAwareGAT
# from basic.trainer import BasicGATTrainer, analyze_dataset


# def main():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     batch_size_raw_data = 16
#     batch_size_trainer = 8
#     learning_rate = 0.001
#     num_epochs = 1500
    
#     embedding_dim = 32
#     keypoint_preprocessor = KeypointPreprocessor(embedding_dim=embedding_dim, device=device)
    
#     print("\n--- Starting Data Preprocessing (this might take a while) ---")
#     train_loader_raw = create_coco_dataloader(
#         split='train',
#         batch_size=batch_size_raw_data,
#         shuffle=False,
#         num_workers=4
#     )
    
#     processed_train_graphs = []
#     for i, batch in enumerate(train_loader_raw):
#         pyg_graphs = keypoint_preprocessor.process_batch(batch)
#         processed_train_graphs.extend(pyg_graphs)
#         if (i + 1) % 100 == 0:
#             print(f"Processed {i + 1} raw batches, total graphs: {len(processed_train_graphs)}")

#     print(f"\nFinished preprocessing. Total training graphs: {len(processed_train_graphs)}")
    
#     print("\nAnalyzing processed dataset statistics...")
#     analyze_dataset(processed_train_graphs)

#     gat_model = DepthAwareGAT(
#         input_dim=4 + embedding_dim,
#         hidden_dim=64,
#         output_dim=32,
#         num_heads=4,
#         dropout=0.6
#     )

#     trainer = BasicGATTrainer(model=gat_model,preprocessor=keypoint_preprocessor, output_dir=settings.output_dir)

#     print("\n--- Starting GAT Training ---")
#     trained_model = trainer.train(
#         dataset=processed_train_graphs,
#         epochs=num_epochs,
#         batch_size=batch_size_trainer,
#         lr=learning_rate
#     )
#     print("\nTraining process completed.")
        
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Import your existing dataloader and settings
from dataloader import create_coco_dataloader
from settings import settings

# Import the new basic2 components
from basic_v2.preprocess import KeypointPreprocessor
from basic_v2.gat import DepthAwareGAT
from basic_v2.trainer import BasicGATTrainer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size_raw_data = 16
    batch_size_trainer = 8
    learning_rate = 0.001
    num_epochs = 1500
    embedding_dim = 32
    
    # Initialize preprocessor
    keypoint_preprocessor = KeypointPreprocessor(embedding_dim=embedding_dim, device=device)
    
    print("\n--- Starting Data Preprocessing (this might take a while) ---")
    train_loader_raw = create_coco_dataloader(
        # split='train',
        split='val',
        batch_size=batch_size_raw_data,
        shuffle=False,
        num_workers=4
    )
    
    # Process data into mixed graphs
    processed_train_graphs = []
    for i, batch in enumerate(train_loader_raw):
        pyg_graphs = keypoint_preprocessor.process_batch(batch)
        processed_train_graphs.extend(pyg_graphs)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} raw batches, total graphs: {len(processed_train_graphs)}")
    
    print(f"\nFinished preprocessing. Total training graphs: {len(processed_train_graphs)}")
    
    # Initialize GAT model
    gat_model = DepthAwareGAT(
        input_dim=4 + embedding_dim,  # [x, y, depth, confidence] + joint embeddings
        hidden_dim=64,
        output_dim=32,
        num_heads=4,
        dropout=0.6
    )
    
    # Initialize trainer
    trainer = BasicGATTrainer(
        model=gat_model,
        preprocessor=keypoint_preprocessor,
        output_dir="outputs/basic2"
    )
    
    print("\n--- Starting GAT Training ---")
    trained_model = trainer.train(
        dataset=processed_train_graphs,
        epochs=num_epochs,
        batch_size=batch_size_trainer,
        lr=learning_rate
    )
    
    print("\nTraining process completed.")
    
    # Save the trained model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'preprocessor_embeddings': keypoint_preprocessor.joint_embeddings.state_dict(),
        'config': {
            'input_dim': 4 + embedding_dim,
            'hidden_dim': 64,
            'output_dim': 32,
            'num_heads': 4,
            'dropout': 0.6,
            'embedding_dim': embedding_dim
        }
    }, "outputs/basic2/trained_model.pth")
    
    print("Model saved to outputs/basic2/trained_model.pth")

if __name__ == "__main__":
    main()