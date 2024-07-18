import torch
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionNetwork
from utils.sampling import sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = AutoregressiveModel().to(device)
    diffusion_model = DiffusionNetwork().to(device)
    
    model.load_state_dict(torch.load("./autoregressive_model_epoch5.pth"))
    diffusion_model.load_state_dict(torch.load("./diffusion_model_epoch5.pth"))
    
    samples = sample(model, diffusion_model, device)
    print(samples)

if __name__ == "__main__":
    main()
