import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image

class dataset(Dataset):
    def __init__(self, dataset_path, label_names):
        self.dataset_path = dataset_path
        
        self.images_path = os.path.join(self.dataset_path, "images")
        self.file_names = [ name[:-4] for name in os.listdir(self.images_path+"/rgb") ]
        
        self.labels_path = os.path.join(self.dataset_path, "labels")
        
        self.label_names = label_names
        
    def __len__(self):
        return len(self.file_names)
        
        
    def __getitem__(self, idx):
        
        file_name = self.file_names[idx]
        
        rgb_path = os.path.join(self.images_path, "rgb", file_name+".jpg")
        nir_path = os.path.join(self.images_path, "nir", file_name+".jpg")
        
        rgb_image = read_image(rgb_path).float()
        nir_image = read_image(nir_path).float()
        
        #4 channel image: R,G,B,NIR
        final_4d_image = torch.cat((rgb_image, nir_image))
        
        label = []
        for label_name in self.label_names:
            label_path = os.path.join(self.labels_path, label_name, file_name+".png")
            label.append(read_image(label_path)/255)
        
        final_4d_image = transforms.Normalize((0.485, 0.456, 0.406, 0.449), (0.229, 0.224, 0.225, 0.226))(final_4d_image)
        final_4d_image = torch.Tensor(final_4d_image)
    
        label = torch.stack(label)
        label = torch.squeeze(label)
        
        return final_4d_image, label