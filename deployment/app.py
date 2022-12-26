import json
import shutil
import os
import imageio
import numpy as np

from util import capFrame, extract_face, get_filenames
from flask import Flask, flash, redirect, render_template, request


import tensorflow as tf
from tensorflow.keras import backend as K
from efficientnet.tfkeras import EfficientNetB6
from tensorflow.keras.models import load_model




app = Flask(__name__ , template_folder="templates", static_folder="static")
key = os.urandom(24)
app.config['SECRET_KEY'] = key



# Frame path
frame = 'C:\\Users\\tooch\\vsproject\\Code\\frame'
      
# Face path
face = 'C:\\Users\\tooch\\vsproject\\Code\\face'

# Video path
video_dir = 'C:\\Users\\tooch\\vsproject\\Code\\videos'
         
         
@app.route('/', methods=['GET'])
def title():
    return render_template('index.html')
  

@app.route('/', methods = ['POST'])
def predict():

      
      try:   
      # creating a folder named videos
        if not os.path.exists('videos'):
          os.makedirs('videos')
     # if not created then raise error
      except OSError:
         print ('Error: Creating directory of videos videos')
         
  
    
      video = request.files['video']
      
      video_path = './videos/' + video.filename
    
    
      # Save uploaded video to video path
      video.save(video_path)
      print(video_path)
    
      # Video path
      cap_path='C:\\Users\\tooch\\vsproject\\Code\\videos\\' + video.filename
      
      # Frame path
      frame_path = 'C:\\Users\\tooch\\vsproject\\Code\\frame\\*'
      
      # Face path
      result_path = 'C:\\Users\\tooch\\vsproject\\Code\\face\\*'
      
      # Extract frame from uploaded video
      capFrame(cap_path)
      
      # Get frame directory
      face_path = get_filenames(frame_path)
      
      # Extract face from frame directory
      extract_face(face_path, required_size=(320, 320))
      
      # Get face dir
      pa = get_filenames(result_path)
      
      for i in range(len(pa)):
            # load image from file
       pixels = imageio.imread(pa[i])
      
      # Predict image and print out result
      # prediction = detect(pixels)
      # print(prediction)
      
    
      
      # loading model
      checkpoint_filepath = 'C:\\Users\\tooch\\vsproject\\Code\\data\\Model'

      best_model = load_model(os.path.join(checkpoint_filepath, 'model.h5'))
      
      face_dir = np.array(pixels)
      
      face_dir = face_dir.reshape(-1,320,320,3)
      
      preds =  best_model.predict(face_dir)
      print(preds)
      for i in preds:
          if i <= 0.49:
            output = 'FAKE'
          
          else:
              output = 'REAL'
    
          classification = '(%.2f%%)' % (preds[0]*100)
          print(classification)
          
      data = output
      data = json.dumps(data)
    
      
      shutil.rmtree(frame)
      shutil.rmtree(face)
      shutil.rmtree(video_dir)
    
        
      return render_template('index.html', preds = data, confidence = classification)
    
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    