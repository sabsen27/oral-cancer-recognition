import torch
import torchvision.transforms as transforms
import os
import json
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from lightly.transforms.moco_transform import MoCoV2Transform

class OralMOCODataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform2 = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            MoCoV2Transform(input_size=224)
        ])
        self.transform = transform

        with open(images, "r") as f:
            self.dataset = json.load(f)

        with open("data/dataset.json", "r") as f:
            self.contrastive_data = json.load(f)

        self.annotations = dict()
        for annotation in self.dataset["annotations"]:
            self.annotations[annotation["image_id"]] = annotation

        self.categories = dict()
        for i, category in enumerate(self.dataset["categories"]):
            self.categories[category["id"]] = i

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        image = self.dataset["images"][idx] 
        annotation = self.annotations[image["id"]]
        
        image_id = image["id"]
        image_name = image["file_name"]
        
        image_path = os.path.join(os.path.dirname(self.images), "oral1", image["file_name"])
        image2 = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform2(image2)
            image2 = self.transform(image2)
            
        category = self.categories[annotation["category_id"]]

        return image, category, image_id, image_name, image2
