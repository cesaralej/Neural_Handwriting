# Neural Network from Scratch to Predict Handwritten Numbers

https://neuralhandwriting.streamlit.app

## Overview

This repository contains the source code for a Streamlit app and a neural network trained on the MNIST Dataset.

## Deployment Instructions

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk) installed
- Google Cloud Project created
- Required libraries installed (`pip install -r requirements.txt`)

### Configuration

1. Set up your Google Cloud SDK by running:

   ```bash
   gcloud init
   ```

2. Deploy the app to Google App Engine:

   ```bash
   gcloud app deploy
   ```

3. Access your app using the provided URL after a successful deployment.

### Local Development

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app locally:

   ```bash
   streamlit run app.py
   ```

3. Access the app in your browser at `http://localhost:8501`.

## App Structure

- **app.py:** Main Streamlit app script.
- **training_neural_network.ipynb:** Script to train the neural network
- **app.yaml:** Configuration file for Google App Engine.
- **requirements.txt:** List of Python dependencies.

## Feedback and Contributions

Feel free to provide feedback, report issues, or contribute to the project. We welcome your input!
