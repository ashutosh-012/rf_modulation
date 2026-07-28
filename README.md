# RF Signal Modulation Classification Pipeline

I was exploring radio frequency signals recently and thinking about how they degrade over the air due to noise and interference. I got curious if deep learning could identify the underlying modulation schemes even when the signals are buried in heavy noise. I decided to build this project to find out, using the RadioML 2016 dataset to see how well different neural networks could classify the signals.

## Project Overview

This is a complete pipeline for Automatic Modulation Classification. It handles everything from downloading the raw data to training models and deploying them. The goal was to take raw IQ data, filter out the completely undetectable noise floors (anything below SNR -4 dB), and see how accurately different architectures could classify the remaining valid signals.

## Architectures Explored

During my exploration, I built and tested several different types of architectures to see how they would handle the time-series data:

1. Basic CNN: A simple convolutional neural network to establish a baseline for extracting spatial features from the IQ data.
2. ResNet: A deeper residual network designed to see if skip connections would help the model learn more complex representations without vanishing gradients.
3. CLDNN: A Convolutional LSTM Deep Neural Network. This architecture combines CNNs to extract the physical shape of the waves with LSTMs to track how those waves change over time.
4. CLDNN with Squeeze-and-Excitation: I added SE blocks to the CLDNN to act as an attention mechanism. This allowed the network to dynamically weight the importance of different features, which ultimately yielded the best performance.

## Running the Pipeline

If you want to run this yourself, here is how I set it up:

1. Clone the repository and navigate into it.
2. Add your Kaggle credentials to a .env file in the root directory (KAGGLE_USERNAME and KAGGLE_KEY).
3. Install the dependencies using pip install hydra-core kaggle python-dotenv timm fastapi uvicorn onnxruntime.

To run the automated pipeline:

python scripts/prepare_data.py
python scripts/train.py model=cldnn_se training.epochs=50
python scripts/export_onnx.py model=cldnn_se

## Evaluation and Results

To evaluate the model and see the accuracy across different Signal-to-Noise Ratios, you can run:

python scripts/evaluate_model.py model=cldnn_se

I found that the CLDNN-SE model was able to achieve around 85 to 90 percent accuracy consistently on the detectable signals (SNR greater than 0 dB). 

## Deployment

To make the model actually usable, I added a deployment script that serves the exported ONNX model via a local REST API.

python src/deployment/server.py

This opens up a server on port 8000. You can test it by sending a POST request to the /predict endpoint with a list of 256 IQ float values, and it will return the classified modulation type.
