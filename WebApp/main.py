
### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt



### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response, stream_with_context, render_template_string
from werkzeug.utils import secure_filename

### Audio imports ###
from library.speech_emotion_recognition import *

### Video imports ###
from library.video_emotion_recognition import *


import tempfile
import time
import math

# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = './Upload'


################################## Home Page #######################################


# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')




############################### VIDEO INTERVIEW ################################


# Read the overall dataframe before the user starts to add his own data
df = pd.read_csv('static/js/db/histo.txt', sep=",")

# Video interview template
@app.route('/video', methods=['POST'])
def video() :
    # Display a warning message
    
    return render_template('video.html')

# Display the video flow (face, landmarks, emotion)
@app.route('/video_1')
def video_1() :
    try :
        # Response is used to display a flow of information
        # file = request.files["file"]
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("$$$$$$$$$$$$$$$$$$$", request.args)
        filename = request.args["file"]
        print("#################", filename)
        
        return Response(gen(os.path.join(app.config['UPLOAD_FOLDER'], filename)),mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        print("##########################################" , e)
        return Response("error")

@app.route('/upload_video', methods=['POST'])
def upload_video() :
    try :
        # Response is used to display a flow of information
        file = request.files["file"]
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        return render_template("video.html", file = filename)

    except Exception as e:
        print("##########################################" , e)
        return Response("error")


# Dashboard
@app.route('/video_dash', methods=("POST", "GET"))
def video_dash():
    
    # Load personal history
    df_2 = pd.read_csv('static/js/db/histo_perso.txt')


    def emo_prop(df_2) :
        return [int(100*len(df_2[df_2.density==0])/len(df_2)),
                    int(100*len(df_2[df_2.density==1])/len(df_2)),
                    int(100*len(df_2[df_2.density==2])/len(df_2)),
                    int(100*len(df_2[df_2.density==3])/len(df_2)),
                    int(100*len(df_2[df_2.density==4])/len(df_2)),
                    int(100*len(df_2[df_2.density==5])/len(df_2)),
                    int(100*len(df_2[df_2.density==6])/len(df_2))]

    emotions = ["Angry", "Disgust", "Fear",  "Happy", "Sad", "Surprise", "Neutral"]
    emo_perso = {}
    emo_glob = {}

    for i in range(len(emotions)) :
        emo_perso[emotions[i]] = len(df_2[df_2.density==i])
        emo_glob[emotions[i]] = len(df[df.density==i])

    df_perso = pd.DataFrame.from_dict(emo_perso, orient='index')
    df_perso = df_perso.reset_index()
    df_perso.columns = ['EMOTION', 'VALUE']
    df_perso.to_csv('static/js/db/hist_vid_perso.txt', sep=",", index=False)

    df_glob = pd.DataFrame.from_dict(emo_glob, orient='index')
    df_glob = df_glob.reset_index()
    df_glob.columns = ['EMOTION', 'VALUE']
    df_glob.to_csv('static/js/db/hist_vid_glob.txt', sep=",", index=False)

    emotion = df_2.density.mode()[0]
    emotion_other = df.density.mode()[0]

    def emotion_label(emotion) :
        if emotion == 0 :
            return "Angry"
        elif emotion == 1 :
            return "Disgust"
        elif emotion == 2 :
            return "Fear"
        elif emotion == 3 :
            return "Happy"
        elif emotion == 4 :
            return "Sad"
        elif emotion == 5 :
            return "Surprise"
        else :
            return "Neutral"


    
    return render_template('video_dash.html', emo=emotion_label(emotion), emo_other = emotion_label(emotion_other), prob = emo_prop(df_2), prob_other = emo_prop(df))


############################### AUDIO INTERVIEW ################################

# Audio Index
@app.route('/audio_index', methods=['POST'])
def audio_index():


    
    return render_template('audio.html', display_button=False)



# Audio Emotion Analysis
@app.route('/audio_dash', methods=("POST", "GET"))
def audio_dash():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    print("#################", filename)
    
    # Sub dir to speech emotion recognition model
    model_sub_dir = os.path.join('Models', 'audio.hdf5')

    # Instanciate new SpeechEmotionRecognition object
    SER = speechEmotionRecognition(model_sub_dir)

    # Voice Record sub dir
    rec_sub_dir = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Predict emotion in voice at each time step
    step = 1 # in sec
    sample_rate = 16000 # in kHz
    
    emotions, timestamp = SER.predict_emotion_from_file(rec_sub_dir, chunk_step=step*sample_rate)

    # Export predicted emotions to .txt format
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions.txt"), mode='w')
    SER.prediction_to_csv(emotions, os.path.join("static/js/db", "audio_emotions_other.txt"), mode='a')

    # Get most common emotion during the interview
    major_emotion = max(set(emotions), key=emotions.count)

    # Calculate emotion distribution
    emotion_dist = [int(100 * emotions.count(emotion) / len(emotions)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df = pd.DataFrame(emotion_dist, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df.to_csv(os.path.join('static/js/db','audio_emotions_dist.txt'), sep=',')

    # Get most common emotion of other candidates
    df_other = pd.read_csv(os.path.join("static/js/db", "audio_emotions_other.txt"), sep=",")

    # Get most common emotion during the interview for other candidates
    major_emotion_other = df_other.EMOTION.mode()[0]

    # Calculate emotion distribution for other candidates
    emotion_dist_other = [int(100 * len(df_other[df_other.EMOTION==emotion]) / len(df_other)) for emotion in SER._emotion.values()]

    # Export emotion distribution to .csv format for D3JS
    df_other = pd.DataFrame(emotion_dist_other, index=SER._emotion.values(), columns=['VALUE']).rename_axis('EMOTION')
    df_other.to_csv(os.path.join('static/js/db','audio_emotions_dist_other.txt'), sep=',')

    # Sleep
    time.sleep(0.5)

    return render_template('audio_dash.html', emo=major_emotion, emo_other=major_emotion_other, prob=emotion_dist, prob_other=emotion_dist_other)


 

if __name__ == '__main__':
    
    app.run(debug=True)
