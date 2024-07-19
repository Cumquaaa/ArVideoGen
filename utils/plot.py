import matplotlib.pyplot as plt
import os

def plot_losses(train_losses, output_dir='./logs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.title('Training Loss per Iter')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'train_losses.png'))
    plt.close()
