## Imports

# Useful
import os
import math
import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot as plt
import sys

# Video
import cv2
from pytube import YouTube

# Audio
from pydub import AudioSegment
import contextlib
import wave
import subprocess

# Server API
from flask import Flask, request, abort
import requests
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # for disable GPU
##################################################################################################

# # Globals
# Video Globals
FRAMES_PER_SECOND = 24
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
SLIDING_WINDOW = 16

# Audio Globals
AUTOTUNE = tf.data.AUTOTUNE
AUDIOS_FOLDER = 'audio'
SECOND = 1000

# Paths
CHECKPOINT_FOLDER = 'checkpoints'
TEST_FOLDER = 'videos'

AUDIO_FORMAT = "wav"
AUDIO_FILE_TYPE = '.wav'

# Server 
SERVER_URL = 'http://10.10.248.106:8080/censored'

# # Models
rgb_model = load_model(CHECKPOINT_FOLDER + '/rgb/model_25_0.72.h5')
flow_model = load_model(CHECKPOINT_FOLDER + '/flow/model_28_0.67.h5')
# combined_rgb_flow_model = load_model(CHECKPOINT_FOLDER+'/combined/31-05-2021_20-14-31/model_01_1.00.h5')
audio_model = load_model(CHECKPOINT_FOLDER + '/audio/ep-80-AudioModel-20_26_49-05-24-2021.h5')

# ## Audio labels
audio_labels = ['Screaming', 'Explosion', 'Fireworks', 'Gunshot_gunfire', 'Civil_defense_siren', 'Music/Talking']

# ## Video labels
video_dataset_labels = {0: 'NonViolence',
                        1: 'Fighting',
                        2: 'Shooting',
                        3: 'Riot/Crowded',
                        4: 'Abuse',
                        5: 'Car accident',
                        6: 'Explosion/Fire/Smoke'}
#######
import traceback
from werkzeug.wsgi import ClosingIterator

class AfterThisResponse:
    def __init__(self, app=None):
        self.callbacks = []
        if app:
            self.init_app(app)

    def __call__(self, callback):
        self.callbacks.append(callback)
        return callback

    def init_app(self, app):
        # install extensioe
        app.after_this_response = self

        # install middleware
        app.wsgi_app = AfterThisResponseMiddleware(app.wsgi_app, self)

    def flush(self):
        try:
            for fn in self.callbacks:
                try:
                    fn()
                except Exception:
                    traceback.print_exc()
        finally:
            self.callbacks = []

class AfterThisResponseMiddleware:
    def __init__(self, application, after_this_response_ext):
        self.application = application
        self.after_this_response_ext = after_this_response_ext

    def __call__(self, environ, start_response):
        iterator = self.application(environ, start_response)
        try:
            return ClosingIterator(iterator, [self.after_this_response_ext.flush])
        except Exception:
            traceback.print_exc()
            return iterator
##################################################################################################
# # Utils
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError:
            print(f"Can't create destination directory {folder_path}!")

##################################################################################################
# # Audio Utils
def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, )  # converted to mono channel
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this 
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform):
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([250000] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)

    spectrogram = tf.abs(spectrogram)

    return spectrogram


def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    spectrogram = tf.expand_dims(spectrogram, -1)
    return spectrogram


def preprocess_audio_files(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
    return output_ds


def convert_mp4_to_wav(file_path):
    # Convert mp4 to wav
    src = file_path

    filename = '.'.join(os.path.basename(file_path).split('.')[:-1])
    dest = TEST_FOLDER +'/'+ filename + AUDIO_FILE_TYPE
    if not os.path.exists(dest):
        command = "ffmpeg -i '" + src + "' -ac 2 -f wav '" + dest + "'"
        subprocess.call(command, shell=True)


def video_to_wav_splits(filename, seconds_to_split=5):
    time_to_split = seconds_to_split * SECOND

    # Split the file to parts
    duration = 0
    file_path = f'{TEST_FOLDER}/{filename}{AUDIO_FILE_TYPE}'
    audio_file = AudioSegment.from_wav(file_path)

    audio_parts_path = f'{TEST_FOLDER}/parts/{filename}'
    ensure_folder_exists(audio_parts_path)

    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)

    for i in range(int(duration / seconds_to_split)):
        start_time = i * time_to_split
        end_time = start_time + time_to_split
        newAudio = audio_file[start_time:end_time]

        # Exports to a wav file in the current path.
        newAudio.export(audio_parts_path + '/' +
                        str(start_time / SECOND) + '-' +
                        str(end_time / SECOND) + AUDIO_FILE_TYPE,
                        format=AUDIO_FORMAT)

##################################################################################################
# ## Video Utils

#### Get Farneback Optical flow
def get_optical_flow(video_frames):
    """
    get_optical_flow -
    Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the
     movement of object or camera. It is 2D vector field where each vector is a displacement vector showing
      the movement of points from first frame to second.
    :param video_frames: the input video with shape of [frames,height,width,channel]. dtype=np.array
    :return:  flows: the optical flows numpy array, with the shape of [frames,height,width,channel]
    """
    gray_frames = []
    flows = []

    for i in range(len(video_frames)):
        img_float32 = np.float32(video_frames[i])
        gray_frame = cv2.cvtColor(img_float32, cv2.COLOR_RGB2GRAY)
        gray_frames.append(np.reshape(gray_frame, (FRAME_HEIGHT, FRAME_WIDTH, 1)))

    for i in range(0, len(gray_frames) - 1):
        # calculate optical flow between each pair of frames
        flow = cv2.calcOpticalFlowFarneback(prev=gray_frames[i], next=gray_frames[i + 1],
                                            flow=None, pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3, poly_n=5,
                                            poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

        # normalize each component in optical flow
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        # Add into list 
        flows.append(flow)

    # Padding the last frame as empty array
    return np.array(flows)


#### Get video frames
def video_to_frames(video_file_path: str, rescale=None, fps=FRAMES_PER_SECOND):
    """
 Convert a video file to individual frames
 """
    if os.path.exists(video_file_path) is None:
        print(f'Error: file path - {video_file_path} not found')
        return

    # Load video capture stream
    cap = cv2.VideoCapture(video_file_path)
    count = 0
    frames = []

    while cap.isOpened():
        frame_id = cap.get(1)  # current frame number
        success, frame = cap.read()  # if the frame is read correctly, it will be True
        if not success:
            break
        if frame is None:
            print(f'frame is none: {frame_id}')

        # We will save every fps that we defined
        if frame_id % math.floor(fps) == 0 or fps == 1:
            frame = cv2.resize(frame, (FRAME_HEIGHT, FRAME_WIDTH))  # Resize pixels
            frame = crop_center_square(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.reshape(FRAME_HEIGHT, FRAME_WIDTH, 3)

            if rescale:
                frame = frame * rescale

            frames.append(frame)

        count += 1

    # When everything is done, release the capture
    cap.release()

    return np.array(frames)


#### Video to npy array
def video_to_npy(video_file_path, fps=FRAMES_PER_SECOND):
    video_frames = video_to_frames(video_file_path=video_file_path, fps=fps)
    flows = get_optical_flow(video_frames)

    return video_frames, flows


def select_frames(frames, index_slide, sliding_window=SLIDING_WINDOW):
    """
  Select a certain number of frames determined by the number (frames_per_video)
  :param index_slide:
  :param frames: list of frames
  :param sliding_window: number of frames to select
  :return: selection of frames
  """

    if index_slide * sliding_window + sliding_window > len(frames):
        frames_before_pad = frames[index_slide * sliding_window:]
        t = sliding_window - len(frames_before_pad)
        target = np.zeros((t, frames.shape[1], frames.shape[2], frames.shape[3]))
        frames_padded = np.append(frames_before_pad, target, axis=0)
        return frames_padded

    frames_selected = frames[index_slide * sliding_window:][:sliding_window]
    return frames_selected


def crop_center_square(frame):
    y, x, c = frame.shape
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def get_seconds(filepath, curr_fps, curr_sliding_window, index):
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS)

    sec = (index * curr_fps * curr_sliding_window) / fps

    return int(math.ceil(sec))


def get_video_duration(filepath):
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return int(math.floor(frame_count / fps))


def show_sample_frames(frames, row_width=22, row_height=5):
    rows = len(frames)

    index = 1
    plt.figure(figsize=(row_width, row_height * rows))
    for batchid, image in enumerate(frames):
        plt.subplot(rows, 6, index)
        plt.imshow(image)
        plt.axis('off')
        index += 1
    plt.show()


def normalize(arr, rescale=1 / 255):
    if (arr.max() > 1):
        arr = arr / 255
    return arr


##################################################################################################

# # Predict
# ## Predict rgb+flow model
def predict_by_video(video_path, fps=5, sliding_window=SLIDING_WINDOW, is_print=True,
                     rescale=True, print_images=True, is_combined=False):
    rgb, flow = video_to_npy(video_path, fps=fps)
    predictions = {}

    if rescale:
        rgb = normalize(rgb)

    for i in range(math.ceil(len(rgb) / sliding_window)):
        # RGB prediction
        rgb_selected = select_frames(rgb, i, sliding_window=sliding_window)
        rgb_expand = tf.expand_dims(rgb_selected, axis=0)

        if print_images:
            print(i)
            show_sample_frames(rgb_selected)

        # Optical flow prediction
        flow_selected = select_frames(flow, i)
        flow_expand = tf.expand_dims(flow_selected, axis=0)

        if is_combined:
            # rgb_logits = combined_rgb_flow_model.predict([rgb_expand, flow_expand])
            print(rgb_logits[0])
            sample_logits = rgb_logits

        else:
            rgb_logits = rgb_model.predict(rgb_expand)
            flow_logits = flow_model.predict(flow_expand)

            # Calc predictions
            sample_logits = rgb_logits + flow_logits

        # produce sigmoid output from model logit for class probabilities
        sample_logits = sample_logits[0]  # we are dealing with just one example

        sample_predictions = [video_dataset_labels[idx] for
                              idx, current_prediction in enumerate(sample_logits) if current_prediction > 0.114]
        if video_dataset_labels[0] in sample_predictions and len(sample_predictions) > 1 and sample_logits[0] < 1.8:
            sample_predictions.pop(0)
        else:
            sample_predictions = [video_dataset_labels[0]]
        #     sorted_indices = np.argsort(sample_predictions)[::-1]

        predictions[get_seconds(video_path, fps, sliding_window, i)] = {'predictions': sample_logits,
                                                                        'predict_labels': sample_predictions}

        if is_print:
            print(sample_predictions)

            for index in range(len(sample_logits)):
                print(f'{get_seconds(video_path, fps, sliding_window, i)} -->', sample_logits[index],
                      video_dataset_labels[index])
            print("--------")

    return predictions, video_dataset_labels


# ## Predict audio
def predict_by_audio(file_path, sliding_window=16):
    # TODO: return the splits and not saved them..
    # Preprocess audio - convert to wav file and splits the file
    filename = '.'.join(os.path.basename(file_path).split('.')[:-1])

    convert_mp4_to_wav(file_path)
    video_to_wav_splits(filename, sliding_window)

    audio_parts_path = f'{TEST_FOLDER}/parts/{filename}'
    predictions = {}

    for audio_part_name in os.listdir(audio_parts_path):
        if audio_part_name != '.ipynb_checkpoints':
            full_file_name = os.path.join(audio_parts_path, audio_part_name)
            sample_ds = preprocess_audio_files([full_file_name])

            for spectrogram in sample_ds.batch(1):
                prediction = audio_model.predict(spectrogram)

                audio_part_sec = os.path.splitext(audio_part_name)[0].split('-')[0].split('.')[0]

                predictions[audio_part_sec] = prediction[0]
    return predictions, audio_labels


# # Get New video
def download_video_from_youtube(youtube_id):
    # download video
    yt = YouTube(f'https://www.youtube.com/watch?v={youtube_id}')
    yt.streams.first().download(TEST_FOLDER)
    os.rename(f'{TEST_FOLDER}/{yt.streams.first().default_filename}', f'{TEST_FOLDER}/{youtube_id}.mp4')


def get_video_and_audio_prediction(file_path, audio_sliding_window=5,
                                   video_fps=5, video_sliding_window=16):
    preds = {}
    video_length = get_video_duration(file_path)

    # predict by video
    results_video, video_labels = predict_by_video(file_path, is_print=False, fps=video_fps,
                                                   sliding_window=video_sliding_window, print_images=False)
    print('results_video')
    # predict by audio
    results_audio, audio_labels = predict_by_audio(file_path, sliding_window=audio_sliding_window)

    for i in range(0, video_length - video_fps, video_fps):
        # making video labels
        temp_video_preds = []
        video_preds = []
        for j in range(i, i + video_fps):
            if j in results_video:
                temp_video_preds.append(results_video[j]['predictions'])
            video_preds_sum = [sum(x) for x in zip(*temp_video_preds)]

        for label_index, video_pred in enumerate(video_preds_sum):
            if (video_pred > 0.15 and label_index != 0 and label_index!=3) or (label_index==3 and video_pred > 0.5):
                video_preds.append({"label": video_dataset_labels[label_index], "pred": video_pred})
        if len(video_preds) == 0 and video_preds_sum[0] > 0.15:
            video_preds.append({"label": video_dataset_labels[0], "pred": video_preds_sum[0]})
        # making audio labels
        audio_preds = []
        for label_index, audio_pred in enumerate(results_audio[str(i)]):
            if audio_pred > 0.5 and label_index != 5:
                audio_preds.append({"label": audio_labels[label_index], "pred": str(audio_pred)})
        if len(audio_preds) == 0 and results_audio[str(i)][5] > 0.5:
            audio_preds.append({"label": audio_labels[5], "pred": str(results_audio[str(i)][5])})
        preds[i] = {"video": video_preds, "audio": audio_preds}

    print('done preds')
    return preds

# # API Server
app = Flask(__name__)
AfterThisResponse(app)


@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json(force=True)
    file_path = data['videoPath']

    if not os.path.exists(file_path):
        abort(400, description='file not found :(')

    @app.after_this_response
    def worker():
        print('get video audio prediction' + file_path)
        preds = get_video_and_audio_prediction(file_path)
        filename = '.'.join(os.path.basename(file_path).split('.')[:-1])
        print('send response')

        headers = {'Content-Type': 'application/json'}
        body  = json.dumps({"content": preds, "name": filename})
        response = requests.post(SERVER_URL, headers=headers, data=str(body), verify=False)
        print(response)

    return 'finished', 200

@app.route('/isalive', methods=['GET'])
def test():
  return 'Alive', 200


app.run(host='0.0.0.0', debug=True)
