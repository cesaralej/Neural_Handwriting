# Streamlit App on Google App Engine

https://syllabus-generator-ie.streamlit.app

## Overview

This repository contains the source code for a Streamlit app designed to transform the learning experience for the "Big Data & Artificial Intelligence in Business Strategy" course. The app utilizes generative AI to personalize syllabus content based on user input.

## Deployment Instructions

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk) installed
- Google Cloud Project created
- Streamlit and OpenAI Python libraries installed (`pip install streamlit openai`)

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
   streamlit run Syllabus_generator.py
   ```

3. Access the app in your browser at `http://localhost:8501`.

## App Structure

- **Syllabus_generator.py:** Main Streamlit app script.
- **app.yaml:** Configuration file for Google App Engine.
- **requirements.txt:** List of Python dependencies.

## Feedback and Contributions

Feel free to provide feedback, report issues, or contribute to the project. We welcome your input!
