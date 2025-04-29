# Maiq The Identifier
Maiq the identifer is a facial recongintion model used for security research. The goal of this model is to provide an example 
of an implementation of a facial recongition CNN. There are various tools for setting up the model, training the model, and 
using the model.

## Tools

### videoScraper.py
This is a tool that takes a video and breaks it down into its frames. These frames are exported into a target destination

### labelGen.py
This is a program which will generate the labels file for us. Use -rec for recussive label generation. Need to be a directory 
with directories seperated by subject name

### MaiqNet.py
The model itself

### modelGen.py
This is the trainer for the model

### fileMove.py
This is used to move the files we used in labelGen.py to a target directory. 

### faceData.py
This is the class used for our custom dataset.

### scrollOfIdentification.py
This is the tool that will use the pretrained model and accept images as input. Currently it only accepts jpgs.