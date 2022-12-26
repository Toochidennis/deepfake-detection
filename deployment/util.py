import cv2
import glob
import imageio
import numpy as np

from mtcnn import MTCNN
from PIL import Image
import os
from mtcnn.mtcnn import MTCNN

import tensorflow as tf
from tensorflow.keras import backend as K
from efficientnet.tfkeras import EfficientNetB6
from tensorflow.keras.models import load_model




def capFrame(video):
    
    cam = cv2.VideoCapture(video)
    try:   
      # creating a folder named images
      if not os.path.exists('frame'):
          os.makedirs('frame')
    # if not created then raise error
    except OSError:
      print ('Error: Creating directory of images data')

    count = 0  # count the number of pictures
    frame_interval = 30  # video frame count interval frequency, the paper mentioned 7 frames per video
    frame_interval_count = 0
    currentframe = 1 # start with frame 1

    while(True):
        # reading from frame
        ret,frame = cam.read()
        if ret:
            if frame_interval_count % frame_interval == 0 and currentframe <=7:
                # if video is still left continue creating images, extract up to 7 imgs /per video
                name = './frame/video' + str(currentframe) + '.jpg'
                print('Creating...' + name)
                frame_interval_count += 1
                # writing the extracted images
                cv2.imwrite(name, frame)
                 # increasing counter so that it will show how many frames are created
                currentframe += 1
            else:
                frame_interval_count += 1
        else:
            break
        
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    
    return print('Done')


def extract_face(filename, required_size=(320, 320)):
    errCount = 0
    for i in range(len(filename)):
        # load image from file
        pixels = imageio.imread(filename[i])
        try:
            # creating a folder faces
            if not os.path.exists('face'):
                os.makedirs('face')
        # if not created then raise error
        except OSError:
            print('Error: Creating directory ')

        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # make sure the captured probability > 0.9 and 4 values in box.keys()
        if results:
            if results[0]['confidence'] >= 0.9 and len(results[0]['box']) == 4:
                # extract the bounding box from the first face
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height
                # extract the face
                print("The confidence is " + str(results[0]['confidence']))
                face = pixels[y1:y2, x1:x2]
                # resize pixels to the model size
                image = Image.fromarray(face)
                image = image.resize(required_size)
                # return face_array
                face_array = np.asarray(image)
                name = './face/image' + "_" + str(i) + '.jpg'
                # writing the extracted images
                imageio.imwrite(name, face_array)
                # print(str(i) + "/" + str(len(filename)) + "iterations")
                print('creating...' + name)
            else:
                errCount += 1
                print("confidence < 0.9")
                continue
        else:
            errCount += 1
            print("The face cannot be captured")
            continue
    return errCount


def get_filenames(path):
    filenames = glob.glob(path.format('.jpg'))
    print("Number of files: ", len(filenames))
    #print(filenames)
    
    return filenames


def detect(face):
    
    # loading model
    checkpoint_filepath = 'C:\\Users\\tooch\\vsproject\\Code\\data\\Model'

    best_model = load_model(os.path.join(checkpoint_filepath, 'model.h5'))
    
    face_dir = np.array(face)
    
    face_dir = face_dir.reshape(-1,320,320,3)
    
    preds =  best_model.predict(face_dir)
    print(preds)
    for i in preds:
        if i <= 0.49:
           print("Fake")
          
        else:
            print('Real')
    
    return face, preds

