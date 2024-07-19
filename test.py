import torch
import os
from PIL import Image
import uuid
import numpy as np
from models.autoregressive import AutoregressiveModel
from models.diffusion import DiffusionLoss
from data.prepare_data import get_dataloader
from utils.args import parse_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_generations_folder():
    generations_folder = 'generations'
    if os.path.exists(generations_folder):
        for filename in os.listdir(generations_folder):
            file_path = os.path.join(generations_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def main():
    # Clear generations folder at the start
    clear_generations_folder()

    config = parse_config()
    
    autoregressive_model = AutoregressiveModel(width=config['trans_width'], depth=config['trans_depth']).to(device)
    diffusion_model = DiffusionLoss(depth=config['mlp_depth'], width=config['mlp_width']).to(device)
    
    autoregressive_model.load_state_dict(torch.load("./ckpts/autoregressive_model_epoch5.pth"))
    diffusion_model.load_state_dict(torch.load("./ckpts/diffusion_model_epoch5.pth"))
    train_loader, test_loader = get_dataloader()
    
    # Create generations folder if it does not exist
    if not os.path.exists('generations'):
        os.makedirs('generations')
    
    with torch.no_grad():
        for batch in train_loader:
            images, num = batch
            images = images.to(device)
            z = autoregressive_model(images)
            
            # Generate noise and timestep
            noise = torch.randn_like(images)
            timestep = torch.randint(0, diffusion_model.diffusion.num_timesteps, (images.size(0),), device=images.device).float()
            timestep = timestep.view(-1, 1)
            
            # Sample the reconstructed images using the diffusion model
            x_recon = diffusion_model.sample(z, noise)
            
            # Post-processing: Convert x_recon to image
            for i in range(images.size(0)):
                # Extract the i-th image and corresponding num
                original_img = images[i].squeeze().cpu().detach().numpy()  # Original image
                generated_img = x_recon[i].squeeze().cpu().detach().numpy()  # Generated image
                
                # Scale to [0, 255] and clip values outside this range
                original_img = np.clip(original_img * 0.5 + 0.5, 0, 1)  # Scale to [0, 1]
                original_img = (original_img * 255).astype(np.uint8)  # Scale to [0, 255]
                
                generated_img = np.clip(generated_img * 0.5 + 0.5, 0, 1)  # Scale to [0, 1]
                generated_img = (generated_img * 255).astype(np.uint8)  # Scale to [0, 255]
                
                # Convert NumPy arrays to PIL images (mode='L' for grayscale)
                original_img = Image.fromarray(original_img, mode='L')
                generated_img = Image.fromarray(generated_img, mode='L')
                
                # Resize images to the same height (number of rows)
                min_height = min(original_img.height, generated_img.height)
                original_img = original_img.resize((original_img.width, min_height))
                generated_img = generated_img.resize((generated_img.width, min_height))
                
                # Concatenate images horizontally into a single NumPy array
                concatenated_img = np.concatenate([np.array(original_img), np.array(generated_img)], axis=1)
                
                # Convert NumPy array back to PIL image
                concatenated_img = Image.fromarray(concatenated_img, mode='L')
                
                # Generate a unique ID for the filename
                unique_id = uuid.uuid4().hex
                
                # Save the concatenated image as JPEG
                save_path = os.path.join('generations', f'{num[i]}_{unique_id}.jpg')
                concatenated_img.save(save_path)
                
                print(f'Saved: {save_path}')
            break

if __name__ == "__main__":
    main()
