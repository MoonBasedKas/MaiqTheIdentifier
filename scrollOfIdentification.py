#!/usr/bin/env python3
"""
This file provicdes an API for calling the model. This will load the model and allow it to begin identification on various parameters.

TODO: Image translation to jpg type.
"""


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import sys
import maiqNet
import faceData
import os 
import numpy as np
from PIL import Image

"""
Color information to improve readability.
"""
class color:
    RED = '\033[31m'
    BLACK = "\033[0;30m"
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BROWN = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    LIGHT_GRAY = '\033[0;37m'
    DARK_GRAY = '\033[1;30m'
    LIGHT_RED = '\033[1;31m'
    LIGHT_GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    LIGHT_BLUE = '\033[1;34m'
    LIGHT_PURPLE = '\033[1;35m'
    LIGHT_CYAN = '\033[1;36m'
    LIGHT_WHITE = '\033[1;37m'
    RESET = '\033[0m'

# Labels to be changed
labels = {
    0: "Gordon",
    1: "Scott",
    2: "Guy"
}

# Processes an image for input.
ImageProcessor = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],  
)

def main():
    argv = sys.argv
    temp = ""
    modelpth = "." + os.sep + "maiqTheIdentifier.pth"
    ioMethod = fileIO
    test = "cat.jpg"
    while argv != []:
        temp = argv.pop(0)
        if temp == "-model":
            modelpth = argv.pop(0)
        elif temp == "-fileIO":
            ioMethod = fileIO
        elif temp == "-internetIO":
            ioMethod = internetIO

    print("Loading the model.")
    maiq = maiqNet.neuralNet()
    print("Model has been successfully loaded.")
    optimizer = optim.Adam(maiq.parameters(), lr=0.001)
    maiq.load_state_dict(torch.load(modelpth, weights_only=True))
    file = "." + os.sep + test
    ioMethod(maiq, file)


"""
Asks the great identifier via file IO. A simple offline method but stores much power and 
attackers must usurp maiq's god like control over the device.
"""
def fileIO(maiq : maiqNet.neuralNet, file):
    global labels
    i = ""
    while i != "q":
        i = input("Enter a file: ")
        try:
            fp = Image.open(i)
            # Don't ask why we convert a PIL image to a numpy array back to a PIL image.
            person = np.asarray(fp)
            
            person = ImageProcessor(person)
            person = person.unsqueeze(0)
            print(person.shape)
            with torch.no_grad():
                result = maiq(person)
                # Check dimension 1
                result = torch.argmax(result, dim=1)
                result = result.item()
                print(labels[result])
        except FileNotFoundError:
            if i != "q":
                print(color.RED + "Error | File not found." + color.RESET)



def internetIO(maiq):
    
    return



if __name__ == "__main__":
    main()