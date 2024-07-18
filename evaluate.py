import torch
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionNetwork
from data.prepare_data import get_dataloader
from utils.metrics import dummy_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    model = AutoregressiveModel().to(device)
    diffusion_model = DiffusionNetwork().to(device)
    
    model.load_state_dict(torch.load("./ckpts/autoregressive_model_epoch5.pth"))
    diffusion_model.load_state_dict(torch.load("./ckpts/diffusion_model_epoch5.pth"))
    
    _, test_loader = get_dataloader()
    
    model.eval()
    diffusion_model.eval()
    
    total_metric = 0
    with torch.no_grad():
        for batch in test_loader:
            images, _ = batch
            images = images.to(device)
            
            z = model(images)
            x_recon = diffusion_model(z)
            
            metric = dummy_metric(x_recon, images)
            total_metric += metric
    
    avg_metric = total_metric / len(test_loader)
    print(f"Average Metric: {avg_metric:.4f}")

if __name__ == "__main__":
    evaluate()
