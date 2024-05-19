import os
from flask import Flask, request, jsonify
import numpy as np
import io
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf
from flask_cors import CORS
import logging
from flask_socketio import SocketIO, emit, rooms
import mediapipe as mp
import cv2
import numpy as np
import os
import tempfile 
from pyngrok import ngrok, conf
import threading
import sys
import uuid 

app = Flask(__name__)

# Load the pre-trained model for holistic
mathmodel = tf.keras.models.load_model("models/math.h5")
langmodel = tf.keras.models.load_model("models/lang.h5")
socialmodel = tf.keras.models.load_model("models/genskills.h5")
envmodel = tf.keras.models.load_model("models/yas.h5")
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


#video process start at
# start_frame = -30
start_frame = -90

#holistic functions
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


# Threshold value for prediction
thresholdValue = 0.6

#Answers array
math_answer_array = ['1','2','3','4']
lang_answer_array = ['අ', 'ආ', 'ඇ','ඈ','ඉ','ඊ','උ','ඌ','එ','ඒ']
# env_answer_array = ['අලියා','පූසා','බල්ලා','ලේනා','ඇපල්','අඹ','මැංගුස්','අන්නාසි','ගස','තරුව']
env_answer_array = ['elephant','cat','dog','squirrel','apple','mango','mongoose','pineapple','tree','star']
# social_answer_array = ["ආයුබෝවන්",
# "උපකාර කරන්න",
# "එක්ඟයි",
# "ඔබව සාදරයෙන් පිලිගන්නවා",
# "කුසගින්න",
# "පිපාසය",
# "සුභ පැතුම්",
# "ස්තූතී",
# "හෙලෝ",
# "හොඳයි"]
social_answer_array=['Ayubowan', 'Isthuthi', 'Hello','Upakara_karanna','Ekagai']
@app.route('/detection/math/v2', methods=['POST'])
def math_detection():
    try:
        data = request.get_json()
        image_url = data['image_url']
        actual_answer = data['answer']
        final_answer = True

        # Download the image from the URL
        response = requests.get(image_url)

        if not response.content:
            raise ValueError("Empty image content")
        # Define the path to save the video
        download_folder = './downloads/'
        os.makedirs(download_folder, exist_ok=True)
        video_filename = f"video_{uuid.uuid4().hex[:5]}.mp4" 
        video_path = os.path.join(download_folder,f"video_{uuid.uuid4().hex[:5]}.mp4")
        with open(video_path, 'wb') as f:
            f.write(response.content)

        # Holistic Prediction
        cap = cv2.VideoCapture(video_path)
        sequence = []
        prediction_result = []
        prediction_result_array = []

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                print("start")
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[start_frame:]

                if len(sequence) == 30:
                    res = mathmodel.predict(np.expand_dims(sequence, axis=0))[0]
                    prediction_result = res
                    print(prediction_result) 
                    break

            cap.release()
            cv2.destroyAllWindows()

            prediction_result_array = prediction_result.tolist()
            max_index = np.argmax(prediction_result_array)
            predicted_answer = math_answer_array[max_index]

            if predicted_answer == actual_answer:
                final_answer = True
            else:
                final_answer = False


        # return jsonify("{name:hello}")
        return jsonify({"result": final_answer,"predicted": predicted_answer})

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/detection/lang/v2', methods=['POST'])
def lang_detection():
    try:
        data = request.get_json()
        image_url = data['image_url']
        actual_answer = data['answer']
        final_answer = True;

        # Download the image from the URL
        response = requests.get(image_url)

        if not response.content:
            raise ValueError("Empty image content")
        # Define the path to save the video
        download_folder = './downloads/'
        os.makedirs(download_folder, exist_ok=True)
        video_filename = f"video_{uuid.uuid4().hex[:5]}.mp4" 
        video_path = os.path.join(download_folder,f"video_{uuid.uuid4().hex[:5]}.mp4")
        with open(video_path, 'wb') as f:
            f.write(response.content)

        # Holistic Prediction
        cap = cv2.VideoCapture(video_path)
        sequence = []
        prediction_result = []
        prediction_result_array = []

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = langmodel.predict(np.expand_dims(sequence, axis=0))[0]
                    prediction_result = res
                    print(prediction_result) 
                    break

            cap.release()
            cv2.destroyAllWindows()

            prediction_result_array = prediction_result.tolist()
            max_index = np.argmax(prediction_result_array)
            predicted_answer = lang_answer_array[max_index]

            if predicted_answer == actual_answer:
                final_answer = True
            else:
                final_answer = False

            print("final_answer",final_answer,"predicted_answer",predicted_answer,"actual_answer",actual_answer)
        # return jsonify("{name:hello}")
        return jsonify({"result": final_answer,"predicted": predicted_answer})

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/detection/env/v2', methods=['POST'])
def env_detection():
    try:
        data = request.get_json()
        image_url = data['image_url']
        actual_answer = data['answer']
        final_answer = True;

        # Download the image from the URL
        response = requests.get(image_url)

        if not response.content:
            raise ValueError("Empty image content")
        # Define the path to save the video
        download_folder = './downloads/'
        os.makedirs(download_folder, exist_ok=True)
        video_filename = f"video_{uuid.uuid4().hex[:5]}.mp4" 
        video_path = os.path.join(download_folder,f"video_{uuid.uuid4().hex[:5]}.mp4")
        with open(video_path, 'wb') as f:
            f.write(response.content)

        # Holistic Prediction
        cap = cv2.VideoCapture(video_path)
        sequence = []
        prediction_result = []
        prediction_result_array = []

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-40:]

                if len(sequence) == 40:
                    res = envmodel.predict(np.expand_dims(sequence, axis=0))[0]
                    prediction_result = res
                    print(prediction_result) 
                    break

            cap.release()
            cv2.destroyAllWindows()

            prediction_result_array = prediction_result.tolist()
            max_index = np.argmax(prediction_result_array)
            predicted_answer = env_answer_array[max_index]

            if predicted_answer == actual_answer:
                final_answer = True
            else:
                final_answer = False

            print("final_answer",final_answer,"predicted_answer",predicted_answer,"actual_answer",actual_answer)
        # return jsonify("{name:hello}")
        return jsonify({"result": final_answer,"predicted": predicted_answer})

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 400


@app.route('/detection/social/v2', methods=['POST'])
def social_detection():
    try:
        data = request.get_json()
        image_url = data['image_url']
        actual_answer = data['answer']
        final_answer = True;

        # Download the image from the URL
        response = requests.get(image_url)

        if not response.content:
            raise ValueError("Empty image content")
        # Define the path to save the video
        download_folder = './downloads/'
        os.makedirs(download_folder, exist_ok=True)
        video_filename = f"video_{uuid.uuid4().hex[:5]}.mp4" 
        video_path = os.path.join(download_folder,f"video_{uuid.uuid4().hex[:5]}.mp4")
        with open(video_path, 'wb') as f:
            f.write(response.content)

        # Holistic Prediction
        cap = cv2.VideoCapture(video_path)
        sequence = []
        prediction_result = []
        prediction_result_array = []

        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = socialmodel.predict(np.expand_dims(sequence, axis=0))[0]
                    prediction_result = res
                    print(prediction_result) 
                    break

            cap.release()
            cv2.destroyAllWindows()

            prediction_result_array = prediction_result.tolist()
            max_index = np.argmax(prediction_result_array)
            predicted_answer = social_answer_array[max_index]

            if predicted_answer == actual_answer:
                final_answer = True
            else:
                final_answer = False


            print("final_answer",final_answer,"predicted_answer",predicted_answer,"actual_answer",actual_answer)
        # return jsonify("{name:hello}")
        return jsonify({"result": final_answer,"predicted": predicted_answer})

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
  app.run()

