import json
import torch
import PIL
import argparse
import matplotlib
import numpy as np
import torchvision as tv
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict
from train import setup_nn

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--top', dest="top", type=int, default=5)
    parser.add_argument('--category_names', dest="category_names_path", type=str, default="./cat_to_name.json")
    parser.add_argument('--gpu', dest="gpu", type=bool, default=True)
    args = parser.parse_args()
    
    print('Chosen configuration:')
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        
    image_path = args.image_path
    checkpoint_path = args.checkpoint_path
    top = args.top
    category_names_path = args.category_names_path
       
    # load neural network
    model = load_nn(checkpoint_path)
    
    # load label mapping
    with open(category_names_path, 'r') as f:
        cat_to_name = json.load(f)
    
    # predict image and display probabilities
    predicted_probabilities, predicted_labels = predict(image_path, model)
    image = process_image(image_path)

    predicted_probabilities = np.array(predicted_probabilities[0])
    predicted_labels = np.array(predicted_labels[0])
    print(predicted_probabilities)
    print(predicted_labels)

    # Show image
    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    ax1.show()


    # Do assignments
    assigned_probabilities = np.array(predicted_probabilities)
    assigned_labels = [cat_to_name[str(label+1)] for label in predicted_labels]

    print(assigned_probabilities)
    print(assigned_labels)

    # Show Assignments
    _,ax2 = plt.subplots()
    ticks = np.arange(len(assigned_labels))
    ax2.bar(ticks, assigned_probabilities)
    ax2.set_xticks(ticks = ticks)
    ax2.set_xticklabels(assigned_labels)
    ax2.yaxis.grid(True)

    plt.show()
    
    
def load_nn(checkpoint_path):    
    checkpoint = torch.load(checkpoint_path)
    model,_,_ = setup_nn(checkpoint['input_size'],
                     checkpoint['hidden_sizes'],
                     checkpoint['output_size'],
                     checkpoint['drop_p'],
                     checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = PIL.Image.open(image)
    image_transforms = tv.transforms.Compose([tv.transforms.Resize(255),
                                              tv.transforms.CenterCrop(224),
                                              tv.transforms.ToTensor(),
                                              tv.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                              ])
    tensor_image = image_transforms(pil_image)
    np_image = tensor_image.numpy()
    
    return np_image
        
    
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax
    
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to('cuda')
    
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.to('cuda')
       
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(image)
        
    output_softmaxed = torch.nn.functional.softmax(output.data, dim=1)
    
    return output_softmaxed.topk(topk)
    
        
if __name__ == '__main__':
    main()
