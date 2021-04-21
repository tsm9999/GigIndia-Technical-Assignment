# Video and Audio Sentiment Analyzer
 GigIndia Technical Assignment

## Introduction

This is a multimodal emotion analysis platform to analyze emotions either by uploading a video or an audio file.

![alt text](https://github.com/tsm9999/GigIndia-Technical-Assignment/blob/main/Screenshots/index.png)

# How to run?

WebApp --> `pip3 install requirements.txt` --> `python3 main.py`

# Directory Structure

    .
    Video and Audio Sentiment Analyzer
    ├── ...
    ├── WebApp                    
    │   ├── templates              
    │   ├── library
    |   │   ├── speech_emotion_recognition.py
    |   │   ├── video_emotion_recognition.py
    |   |   └── ...  
    │   ├── static             
    │   ├── main.py
    │   ├── requirements.txt
    |   └── ...  
    ├── Audio 
    |
    ├── Video 
    |
    ├── Screenshots
    |
    ├── Video Input Samples
    |
    ├── Audio Input Samples
    |
    └── ... 
    .
    
# Methodology

The aim of the assignment was to develop a model able to provide sentiment analysis fro video and audio files.

### a. Video Analysis
<br>
1. Video Analysis<img src="https://github.com/tsm9999/Video-and-Audio-Sentiment-Analyzer/blob/main/Screenshots/video_analysis.png" width="1000" height="500">
<br>
<br>
2. Video Sentiment Dashboard<img src="https://github.com/tsm9999/Video-and-Audio-Sentiment-Analyzer/blob/main/Screenshots/video_dashboard.png" width="1000" height="800">
<br>

#### Video Pipeline

The video processing pipeline was built the following way :
- Upload video file.
- Identify the face by Histogram of Oriented Gradients.
- Zoom on the faces.
- Dimension the face to 48 * 48 pixels.
- Make a prediction on the face using our pre-trained model.
- Also identify the number of blinks on the facial landmarks on each picture.


### b. Audio Analysis
<br>
1. Audio Sentiment Dashboard <img src="https://github.com/tsm9999/Video-and-Audio-Sentiment-Analyzer/blob/main/Screenshots/audio_dashboard.png" width="1000" height="500">
 <br>

#### Audio Pipeline

The speech emotion recognition pipeline was built the following way :
- Upload audio .wav file
- Log-mel-spectrogram extraction
- Split spectrogram using a rolling window
- Make a prediction using a pre-trained model


# Resources Used

- https://github.com/speechbrain/speechbrain/
- https://github.com/ankurbhatia24/MULTIMODAL-EMOTION-RECOGNITION
- https://github.com/maelfabien/Multimodal-Emotion-Recognition
- https://github.com/kousik97/Video-Expression-Recognition
- etc..
