import torchvision.datasets as datasets
from torchvision import transforms
from pathlib import Path

# Get the current directory (where the API code is located)
api_dir = Path(__file__).parent

# Define the transformation (optional)
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to match the desired image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Set the dataset directory to the API directory
dataset_dir = api_dir / 'lfw_dataset'  # Creates a folder named 'lfw_dataset' inside the API directory

# Download and load the dataset
lfw_dataset = datasets.LFWPeople(root=dataset_dir, download=True, transform=transform)

# Optional: Access the classes and number of classes
lfw_classes = lfw_dataset.classes
n_classes = len(lfw_classes)

print(f"Number of classes: {n_classes}")
print(f"First 5 classes: {lfw_classes[:5]}")
