import torch
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionLoss
from data.prepare_data import get_dataloader
from utils.args import parse_config
from utils.sampling import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    autoregressive_model = AutoregressiveModel().to(device)
    diffusion_model = DiffusionLoss().to(device)
    
    autoregressive_model.load_state_dict(torch.load("./ckpts/autoregressive_model_epoch5.pth"))
    diffusion_model.load_state_dict(torch.load("./ckpts/diffusion_model_epoch5.pth"))
    
    samples = sample(diffusion_model, device)
    print(samples)

if __name__ == "__main__":
    main()
