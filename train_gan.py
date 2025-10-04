import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
import numpy as np

class Config:
    DATA_DIR = './celeba-dataset/img_align_celeba/img_align_celeba'
    CHECKPOINT_DIR = './checkpoints'
    SAMPLE_DIR = './generated_faces'
    
    IMAGE_SIZE = 64
    BATCH_SIZE = 64
    LATENT_SIZE = 128
    LR = 0.0002
    
    TRAIN_TIME_MINUTES = 10
    CHECKPOINT_INTERVAL = 2
    
    STATS = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.SAMPLE_DIR, exist_ok=True)

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)

device = get_device()

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import glob

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                          glob.glob(os.path.join(root_dir, "*.png"))
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0 

def setup_dataloader():
    print("\nSetting up dataloader...")
    possible_paths = [
        './celeba-dataset/img_align_celeba/img_align_celeba',
        './celeba-dataset/img_align_celeba',
        './img_align_celeba',
        Config.DATA_DIR
    ]

    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            jpg_files = glob.glob(os.path.join(path, "*.jpg"))
            png_files = glob.glob(os.path.join(path, "*.png"))
            if len(jpg_files) > 0 or len(png_files) > 0:
                data_path = path
                break
    
    if data_path is None:
        print("Dataset not found!")
        print("Download from: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        print("Extract to: ./celeba-dataset/")
        print("\nLooking for images in:")
        for path in possible_paths:
            print(f"   - {path}")
        return None
    
    print(f"✓ Found dataset at: {data_path}")
    
    try:
        train_ds = CelebADataset(
            data_path,
            transform=T.Compose([
                T.Resize(Config.IMAGE_SIZE),
                T.CenterCrop(Config.IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(*Config.STATS)
            ])
        )
        
        train_dl = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Dataset loaded: {len(train_ds)} images")
        return DeviceDataLoader(train_dl, device)
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def create_discriminator():
    return nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        nn.Flatten(),
        nn.Sigmoid()
    )

def create_generator():
    return nn.Sequential(
        nn.ConvTranspose2d(Config.LATENT_SIZE, 512, 4, 1, 0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )

def save_checkpoint(epoch, gen, disc, opt_g, opt_d, metrics, name='latest'):
    filepath = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_{name}.pth')
    torch.save({
        'epoch': epoch,
        'generator': gen.state_dict(),
        'discriminator': disc.state_dict(),
        'optimizer_g': opt_g.state_dict(),
        'optimizer_d': opt_d.state_dict(),
        'metrics': metrics
    }, filepath)
    print(f"Checkpoint saved: {filepath}")

def load_checkpoint(name='latest'):
    filepath = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_{name}.pth')
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    return None

def train_discriminator(real_images, gen, disc, opt_d):
    opt_d.zero_grad()
    
    real_preds = disc(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    latent = torch.randn(real_images.size(0), Config.LATENT_SIZE, 1, 1, device=device)
    fake_images = gen(latent)
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = disc(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()
    
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    
    return loss.item(), real_score, fake_score

def train_generator(gen, disc, opt_g, batch_size):
    opt_g.zero_grad()
    
    latent = torch.randn(batch_size, Config.LATENT_SIZE, 1, 1, device=device)
    fake_images = gen(latent)
    
    preds = disc(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    loss.backward()
    opt_g.step()
    
    return loss.item()

def train(train_dl, minutes=10, resume=False):
    print(f"\n{'='*60}")
    print(f"Starting {minutes}-minute training session")
    print(f"{'='*60}\n")
    
    gen = create_generator().to(device)
    disc = create_discriminator().to(device)
    
    opt_g = torch.optim.Adam(gen.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=Config.LR, betas=(0.5, 0.999))
    
    start_epoch = 0
    metrics = {'losses_g': [], 'losses_d': [], 'real_scores': [], 'fake_scores': []}
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            gen.load_state_dict(checkpoint['generator'])
            disc.load_state_dict(checkpoint['discriminator'])
            opt_g.load_state_dict(checkpoint['optimizer_g'])
            opt_d.load_state_dict(checkpoint['optimizer_d'])
            start_epoch = checkpoint['epoch']
            metrics = checkpoint['metrics']
    
    start_time = time.time()
    last_checkpoint_time = start_time
    epoch = start_epoch
    
    try:
        while (time.time() - start_time) < minutes * 60:
            epoch += 1
            epoch_start = time.time()
            
            for real_images, _ in tqdm(train_dl, desc=f'Epoch {epoch}'):
                loss_d, real_score, fake_score = train_discriminator(real_images, gen, disc, opt_d)
                loss_g = train_generator(gen, disc, opt_g, real_images.size(0))
            
            metrics['losses_g'].append(loss_g)
            metrics['losses_d'].append(loss_d)
            metrics['real_scores'].append(real_score)
            metrics['fake_scores'].append(fake_score)
            
            elapsed = (time.time() - start_time) / 60
            print(f"{elapsed:.1f}min | Epoch {epoch} | "
                  f"L_g: {loss_g:.3f} | L_d: {loss_d:.3f} | "
                  f"R: {real_score:.3f} | F: {fake_score:.3f}")
            
            if (time.time() - last_checkpoint_time) > Config.CHECKPOINT_INTERVAL * 60:
                save_checkpoint(epoch, gen, disc, opt_g, opt_d, metrics)
                save_samples(gen, epoch)
                last_checkpoint_time = time.time()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    save_checkpoint(epoch, gen, disc, opt_g, opt_d, metrics)
    save_samples(gen, epoch, show=True)
    
    print(f"\n✓ Training complete! Total epochs: {epoch}")
    return gen, disc, metrics

def denorm(img):
    return img * Config.STATS[1][0] + Config.STATS[0][0]

def save_samples(gen, epoch, num_images=64, show=False):
    gen.eval()
    with torch.no_grad():
        latent = torch.randn(num_images, Config.LATENT_SIZE, 1, 1, device=device)
        fake_images = gen(latent)
        filepath = os.path.join(Config.SAMPLE_DIR, f'epoch_{epoch:04d}.png')
        save_image(denorm(fake_images), filepath, nrow=8, normalize=True)
        print(f"Saved samples: {filepath}")
        
        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(make_grid(denorm(fake_images.cpu()), nrow=8).permute(1, 2, 0))
            plt.axis('off')
            plt.title(f'Generated Faces - Epoch {epoch}')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.SAMPLE_DIR, f'display_epoch_{epoch}.png'))
            plt.show()
    gen.train()

def generate_faces(num_images=16, checkpoint_name='latest'):
    print(f"\nGenerating {num_images} faces...")
    
    checkpoint = load_checkpoint(checkpoint_name)
    if not checkpoint:
        print("No checkpoint found! Train the model first.")
        return
    
    gen = create_generator().to(device)
    gen.load_state_dict(checkpoint['generator'])
    gen.eval()
    
    with torch.no_grad():
        latent = torch.randn(num_images, Config.LATENT_SIZE, 1, 1, device=device)
        fake_images = gen(latent)
        
        plt.figure(figsize=(12, 12))
        plt.imshow(make_grid(denorm(fake_images.cpu()), nrow=4).permute(1, 2, 0))
        plt.axis('off')
        plt.title('AI Generated Human Faces')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.SAMPLE_DIR, 'generated_output.png'), dpi=150)
        plt.show()
        
        print(f"✓ Generated {num_images} faces!")

def generate_with_seed(seed=42, num_images=16):
    torch.manual_seed(seed)
    generate_faces(num_images)

def plot_training_progress(metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(metrics['losses_d'], label='Discriminator', alpha=0.7)
    ax1.plot(metrics['losses_g'], label='Generator', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(metrics['real_scores'], label='Real Score', alpha=0.7)
    ax2.plot(metrics['fake_scores'], label='Fake Score', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Discriminator Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.SAMPLE_DIR, 'training_progress.png'), dpi=150)
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("🎭 Human Faces GAN Training (CelebA Dataset)")
    print("="*60)
    
    train_dl = setup_dataloader()
    
    if train_dl is None:
        print("\nCannot proceed without dataset!")
        print("\nSetup Instructions:")
        print("1. Download: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset")
        print("2. Extract to: ./celeba-dataset/")
        print("3. Run this script again")
    else:
        print("\n" + "="*60)
        print("TRAINING OPTIONS")
        print("="*60)
        print("1. Start NEW training (10 min)")
        print("2. RESUME from checkpoint (10 min)")
        print("3. GENERATE faces from checkpoint")
        print("4. View training progress")
        print("="*60)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            gen, disc, metrics = train(train_dl, minutes=10, resume=False)
            plot_training_progress(metrics)
            
        elif choice == '2':
            gen, disc, metrics = train(train_dl, minutes=10, resume=True)
            plot_training_progress(metrics)
            
        elif choice == '3':
            num = input("Number of faces to generate (default 16): ").strip()
            num = int(num) if num else 16
            generate_faces(num)
            
        elif choice == '4':
            checkpoint = load_checkpoint()
            if checkpoint:
                plot_training_progress(checkpoint['metrics'])
            else:
                print("No checkpoint found!")
        
        else:
            print("Invalid choice!")
    
    print("\n" + "="*60)
    print("✓ Done!")
    print("="*60)