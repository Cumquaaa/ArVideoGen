import torch
import torch.optim as optim
from data.prepare_data import get_dataloader
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionNetwork
from utils.losses import diffusion_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    model = AutoregressiveModel().to(device)
    diffusion_model = DiffusionNetwork().to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(diffusion_model.parameters()), lr=0.001)
    train_loader, test_loader = get_dataloader()
    
    model.train()
    diffusion_model.train()
    
    for epoch in range(5):
        train_loss = 0.0
        for batch in train_loader:
            images, _ = batch
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Autoregressive model predicts a vector z for each token
            z = model(images)
            
            # Denoising diffusion network models the distribution p(x|z)
            x_recon = diffusion_model(z)
            
            # Compute the Diffusion Loss
            loss = diffusion_loss(x_recon, images)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        test_loss = evaluate(model, diffusion_model, test_loader)
        
        print(f"Epoch [{epoch+1}/5], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save checkpoints
        torch.save(model.state_dict(), f"./ckpts/autoregressive_model_epoch{epoch+1}.pth")
        torch.save(diffusion_model.state_dict(), f"./ckpts/diffusion_model_epoch{epoch+1}.pth")

def evaluate(model, diffusion_model, dataloader):
    model.eval()
    diffusion_model.eval()
    
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            images = images.to(device)
            
            # Autoregressive model predicts a vector z for each token
            z = model(images)
            
            # Denoising diffusion network models the distribution p(x|z)
            x_recon = diffusion_model(z)
            
            # Compute the Diffusion Loss
            loss = diffusion_loss(x_recon, images)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    train()
