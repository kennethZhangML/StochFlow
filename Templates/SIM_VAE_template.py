import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the dimensions of the latent space and image space
latent_dim = 128
image_dim = 3 * 64 * 64  # Assuming 3-channel RGB images of size 64x64

# Define the encoder and decoder neural networks
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(image_dim, 512)
        self.fc2_mean = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        mean = self.fc2_mean(x)
        logvar = self.fc2_logvar(x)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, image_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = x.view(x.size(0), 3, 64, 64)  # Reshape to image dimensions
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

batch_size = 64
dataset = torchvision.datasets.YOUR_DATASET(root = 'path/to/your/dataset', transform = transform)
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr = 1e-3)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        
        optimizer.zero_grad()
        reconstructed_data, mean, logvar = vae(data)
        
        # Define the reconstruction loss and the KL divergence term
        reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed_data, data, reduction = 'sum')
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss = Reconstruction loss + KL divergence
        total_loss = reconstruction_loss + kl_divergence
        
        total_loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx + 1} / {len(dataloader)}], Loss: {total_loss.item()/len(data):.4f}')

# Generating new images
num_generated_images = 10
with torch.no_grad():
    z_samples = torch.randn(num_generated_images, latent_dim).to(device)
    generated_images = vae.decoder(z_samples)

# Now you have generated images in the "generated_images" tensor

# You can save and visualize the generated images as needed
