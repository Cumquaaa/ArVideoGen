import torch
import torch.optim as optim
from data.prepare_data import get_dataloader
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionLoss
from utils.args import parse_config
from utils.plot import plot_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    config = parse_config()
    
    model = AutoregressiveModel(width=config['trans_width'], depth=config['trans_depth']).to(device)
    diffusion_loss_module = DiffusionLoss(depth=config['mlp_depth'], width=config['mlp_width']).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(diffusion_loss_module.parameters()), lr=0.001)
    train_loader, test_loader = get_dataloader()
    
    model.train()
    diffusion_loss_module.train()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(5):
        model.train()
        diffusion_loss_module.train()
        
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            images, _ = batch
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Add preprocessing module to linearly project raw pixels to a lower dimension. Ref DiT/FuYu-Flexible Generation.
            # Roughly equals VAE's tokenization step.
            # (256*256, seq_len)*3 => 128*128*? => (32*32)*1024 (bottle-neck)
            # Decrease seq_len, find best patch size.
            # Patch first, then project with common module.
            
            # Ref ImageGPT? AIM - Apple, patchify.
            
            # Autoregressive model predicts a vector z for each token
            # Leave out bos/eos
            z = model(images) # Output dim should be batch_size * seq_len * hidden.
            # Check on mask. Bidirectional attention? Try block inference, for further implementation.
            # Maybe add z-loss?
            
            # Compute the Diffusion Loss
            loss = diffusion_loss_module(z, images) # Concept of noisy-x, see x_t in DiT.
            # Take the loss out, and add linear layers for decoding hidden. Maybe more params for decoder. Ref VAE.
            # Write out all dims and loss before coding.
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_losses.append(loss.item())
            
            if (i + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/5], Iter [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        test_loss = evaluate(model, diffusion_loss_module, test_loader)
        test_losses.append(test_loss)
        
        print(f"Epoch [{epoch+1}/5], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save checkpoints
        torch.save(model.state_dict(), f"./ckpts/autoregressive_model_epoch{epoch+1}.pth")
        torch.save(diffusion_loss_module.state_dict(), f"./ckpts/diffusion_model_epoch{epoch+1}.pth")
    
    plot_losses(train_losses)

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
            
            # Compute the Diffusion Loss
            loss = diffusion_model(z, images)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    train()
