# AI Content Detector
![image](https://github.com/user-attachments/assets/5953cad2-1b5c-433b-9378-ac9feb48b71f)


## Project Description

The AI Content Detector is a tool designed to identify whether a given piece of content (text, image, audio, or video) is AI-generated or human-created. The tool uses a combination of feature extraction, transformer model embeddings, and a hybrid classifier to provide probabilities based on the features and the content provided.

## Purpose

The purpose of this project is to create a comprehensive tool that can analyze various types of content and determine the likelihood of it being AI-generated. This can be useful in various applications such as content moderation, authenticity verification, and detecting AI-generated misinformation.

## How to Use the Tool

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/wylieglover/AI-Content-Detector.git
   cd AI-Content-Detector
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.7 or higher installed. Then, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Tool:**
   You can run the tool using the provided GUI. Simply execute the `main.py` file:
   ```bash
   python main.py
   ```

4. **Select Content:**
   Use the GUI to select the file you want to analyze. The tool will process the content and display the analysis results, including the decision on whether the content is AI-generated or human-created.

## Dependencies

The following dependencies are required to run the AI Content Detector:

- `nltk`
- `numpy`
- `scikit-learn`
- `transformers`
- `tensorflow[and-cuda]`
- `tf-keras`
- `pillow`
- `opencv-python`
- `requests`
- `librosa`
- `python-magic`
- `joblib`
- `StrEnum`
- `spacy`
- `torch`
- `beautifulsoup4`

You can install all dependencies using the `requirements.txt` file provided in the repository.
