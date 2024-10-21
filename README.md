# Emotion Recognition from Facial Expressions

This project demonstrates the process of recognizing emotions from facial expressions using deep learning techniques with PyTorch. We utilize the FER2013 dataset and a pre-trained VGG16 model to classify facial expressions into seven emotion categories.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup and Installation](#setup-and-installation)
4. [Project Structure](#project-structure)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Results](#results)
8. [Usage](#usage)


## Project Overview

This project aims to classify facial expressions into seven emotion categories: Anger, Disgust, Fear, Happy, Sad, Surprise, and Neutral. We use transfer learning with a pre-trained VGG16 model, fine-tuning it on the FER2013 dataset.

## Dataset

We use the FER2013 dataset, which contains 48x48 pixel grayscale images of faces. The faces have been automatically registered so that they are more or less centered and occupy about the same amount of space in each image.

## Setup and Installation

1. Clone this repository
 ```
git clone https://github.com/yassinebourkha/Emotion-Recognition.git
 ```
2. Launch the notebook
 ```
jupyter notebook Emotion_Recognition_from_Facial_Expressions.ipynb
 ```
## Project Structure

- `emotion_recognition.ipynb`: Main Jupyter notebook containing the entire project workflow
- `EmotionRecognition.pth`: Saved model weights after training

## Model Architecture

We use a pre-trained VGG16 model as our base model and modify the classifier layer to output 7 classes corresponding to the emotions in the FER2013 dataset.

## Training

The model is trained for 15 epochs using the Adam optimizer and Cross-Entropy loss function. We use data augmentation techniques like random horizontal flipping to improve model generalization.

## Results

After training, our model achieved:
- Training Accuracy: 62.61%
- Test Accuracy: 58.44%

We provide visualizations of the training process, including loss and accuracy curves, as well as a confusion matrix to analyze the model's performance across different emotion categories.

## Usage

To use the trained model for inference:

1. Load the saved model weights:
   ```python
   model = models.vgg16(pretrained=False)
   model.classifier[6] = nn.Linear(4096, 7)
   model.load_state_dict(torch.load('EmotionRecognition.pth'))
   ```

2. Preprocess your input image and pass it through the model:
   ```python
   # Preprocess your image
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
   input_image = transform(input_image).unsqueeze(0)

   # Get prediction
   with torch.no_grad():
       output = model(input_image)
       _, predicted = torch.max(output, 1)
   ```

