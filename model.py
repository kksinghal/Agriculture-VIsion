import torch
from torch import nn
import torchvision.models as models

model = models.resnet50(pretrained=True)
resnet_layers = list(model.children())

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        model = models.resnet50(pretrained=True)
        
        #Add weights for 4th dimension input
        nir_channel_weight = torch.mean(resnet_layers[0]._parameters["weight"], dim=1, keepdims=True)
        conv1_weights = torch.cat((resnet_layers[0]._parameters["weight"], \
                                                                   nir_channel_weight), dim=1)
        
        conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        conv1.weight = torch.nn.Parameter(conv1_weights)
        
        self.initial_layer = nn.Sequential(
            conv1,
            *resnet_layers[1:4]
        )
        
        #Intentionally using list instead of sequential to store activations of each layer
        self.layers = nn.ModuleList(resnet_layers[4:8])
        
        self.layer_output_conv = nn.ModuleList([
            nn.Conv2d(256, 9, kernel_size=1, stride=1),
            nn.Conv2d(512, 9, kernel_size=1, stride=1),
            nn.Conv2d(1024, 9, kernel_size=1, stride=1)
        ])
        
        self.upsamplers = nn.ModuleList([
            nn.ConvTranspose2d(2048, 9, kernel_size=2, stride=2),
            nn.ConvTranspose2d(18, 9, kernel_size=2, stride=2),
            nn.ConvTranspose2d(18, 9, kernel_size=2, stride=2)
        ])
        
        self.before_pred_convs = nn.ModuleList([
            nn.Conv2d(18, 9, kernel_size=1, stride=1),
            nn.Conv2d(18, 9, kernel_size=1, stride=1),
            nn.Conv2d(18, 9, kernel_size=1, stride=1)
        ])
        
    def forward(self, X):
        predictions = []
        
        out = self.initial_layer(X)
        
        layer_outputs = []

        for layer in self.layers:
            out = layer(out)
            layer_outputs.append(out)
        
        layer_conv_outputs = []
        for i in range(len(layer_outputs)-1):
            layer_conv_outputs.append(self.layer_output_conv[i](layer_outputs[i]))
        
        concat_out = layer_outputs[-1]
        for i in range(len(self.upsamplers)):
            upsampler = self.upsamplers[i]
            upsampler_out = upsampler(concat_out)
            
            concat_out = torch.cat((upsampler_out, layer_conv_outputs[-i-1]), dim=1)
            pred = self.before_pred_convs[i](concat_out)
            predictions.append(pred)
        
        predictions = [ torch.sigmoid(prediction) for prediction in predictions ]
        
        return predictions 
        