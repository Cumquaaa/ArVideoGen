import torch
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionLoss
from data.prepare_data import get_dataloader
from utils.metrics import dummy_metric
from utils.args import parse_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    # Load models
    config = parse_config()
    
    model = AutoregressiveModel(width=config['trans_width'], depth=config['trans_depth']).to(device)
    diffusion_model = DiffusionLoss(depth=config['mlp_depth'], width=config['mlp_width']).to(device)
    
    model.load_state_dict(torch.load("./ckpts/autoregressive_model_epoch5.pth"))
    diffusion_model.load_state_dict(torch.load("./ckpts/diffusion_model_epoch5.pth"))
    
    # Get test data loader
    _, test_loader = get_dataloader()
    
    model.eval()
    diffusion_model.eval()
    
    total_metric = 0
    with torch.no_grad():
        for batch in test_loader:
            images, _ = batch
            images = images.to(device)
            
            # Get the conditioning vector z from the autoregressive model
            z = model(images)
            
            # Generate noise and timestep
            noise = torch.randn_like(images)
            timestep = torch.randint(0, diffusion_model.diffusion.num_timesteps, (images.size(0),), device=images.device).float()
            timestep = timestep.view(-1, 1)
            
            # Sample the reconstructed images using the diffusion model
            x_recon = diffusion_model.sample(z, noise)
            
            # Compute the metric
            metric = dummy_metric(x_recon, images)
            total_metric += metric
    
    avg_metric = total_metric / len(test_loader)
    print(f"Average Metric: {avg_metric:.4f}")

if __name__ == "__main__":
    evaluate()
