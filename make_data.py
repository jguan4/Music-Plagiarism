import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import *

# Configuration
DIAGRAM = 'chroma' # choose between 'mel' (for spectrogram) and 'chroma' (for chroma feature)
DIAGRAM_STR = "melspectrogram" if DIAGRAM == 'mel' else "chroma_feature"
EXCEL_PATH = 'WSdata3.csv'  # Update this path
AUDIO_FILES_PATH = './YouTube_audios_updated/'  # Update this path
SAMPLE_RATE = 48000
TIME_TOLERANCE = 1.0  # in seconds
BPM_DURATION = True # boolean toggle, if true, then clip duration is 8 bars, secs is based on bpm 
SAMPLE_LENGTH = 10  # in seconds
if BPM_DURATION:
    DATASET_ROOT = f"./{DIAGRAM_STR}_preprocessed_dataset/"
else:
    DATASET_ROOT = f"./{DIAGRAM_STR}{SAMPLE_LENGTH}s_dataset/"
SEMITONE_LS = [-5,-4,-3,-2,-1,1,2,3,4,5,6]

# Function to save mel-spectrogram to an image file
def save_mel_spectrogram(audio_clip, file_path):
    # Produce the mel-spectrogram
    S = librosa.feature.melspectrogram(y=audio_clip, sr=SAMPLE_RATE)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Save the mel-spectrogram
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(S_DB, sr=SAMPLE_RATE)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.close()


def save_chroma_feature(audio_clip, file_path):
     # Produce the chroma feature
    S = np.abs(librosa.stft(audio_clip, n_fft=4096))**2
    S = librosa.feature.chroma_stft(y=audio_clip, sr=SAMPLE_RATE)
    chroma = librosa.amplitude_to_db(S, ref=np.max)

    # Save the mel-spectrogram
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(chroma, sr=SAMPLE_RATE)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel-Spectrogram')
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.close()


# Function to process each audio file
def process_audio_files(row):
    original_song_path = os.path.join(AUDIO_FILES_PATH, replace_invalid_char(row['orig Songs Names'])+".mp3")
    sample_song_path = os.path.join(AUDIO_FILES_PATH, replace_invalid_char(row['Sampling Song Names'])+".mp3")

    # load audio
    y, sr = librosa.load(original_song_path, sr=SAMPLE_RATE)
    y_sample, sr_sample = librosa.load(sample_song_path, sr=SAMPLE_RATE)

    # find start time 
    original_start_times = convert_time_to_sec(row['Start of sample in orig song'])
    sample_start_times = convert_time_to_sec(row['Start of Sample'])

    # only consider the first sample right now
    if type(original_start_times) != int:
        original_start_time = original_start_times[0]
    else:
        original_start_time = original_start_times

    if type(sample_start_times) != int:
        sample_start_time = sample_start_times[0]
    else:
        sample_start_time = sample_start_times
    
    # calculate duration to clip
    if BPM_DURATION:
        origin_duration = calculate_eightbars_duration(y, SAMPLE_RATE)
        sample_duration = calculate_eightbars_duration(y_sample, SAMPLE_RATE)
        original_end_time = original_start_time + origin_duration
        sample_end_time = sample_start_time + sample_duration
    else:
        original_end_time = original_start_time + SAMPLE_LENGTH
        sample_end_time = sample_start_time + SAMPLE_LENGTH
    
    # Clip the audio segment
    # ------------------- original song
    start_sample = librosa.time_to_samples(original_start_time, sr=SAMPLE_RATE)
    end_sample = librosa.time_to_samples(original_end_time, sr=SAMPLE_RATE)
    audio_clip_origin = y[start_sample:end_sample]
    # ------------------- sample song
    start_sample_sample = librosa.time_to_samples(sample_start_time, sr=SAMPLE_RATE)
    end_sample_sample = librosa.time_to_samples(sample_end_time, sr=SAMPLE_RATE)
    audio_clip_sample = y_sample[start_sample_sample:end_sample_sample]
    
    # Construct file path
    folder_path = os.path.join(DATASET_ROOT, replace_invalid_char(row['orig Songs Names']))
    os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't exist

    folders = os.listdir(folder_path)
    list_sampletimes = list_of_existing_sampletime(folders)
    if list_sampletimes.any():
        indnum = find_closest_number(list_sampletimes, original_start_time)
        if np.abs(list_sampletimes[indnum]-original_start_time)<=TIME_TOLERANCE:
            start_time_folder_path = os.path.join(folder_path, str(list_sampletimes[indnum]))
        else:
            start_time_folder_path = os.path.join(folder_path, str(original_start_time))
            os.makedirs(start_time_folder_path, exist_ok=True)
    else:
        start_time_folder_path = os.path.join(folder_path, str(original_start_time))
        os.makedirs(start_time_folder_path, exist_ok=True)

    # folder = original_start_time if use_folder is None else use_folder
    file_name = f"{replace_invalid_char(row['Sampling Song Names'])}_{sample_start_time}.png"
    file_path = os.path.join(start_time_folder_path, file_name)
    if not os.path.isfile(file_path):
        if DIAGRAM == 'mel':
            # Save the mel-spectrogram as a PNG image
            save_mel_spectrogram(audio_clip_sample, file_path)
        elif DIAGRAM == 'chroma':
            save_chroma_feature(audio_clip_sample, file_path)

    # save original
    file_name = f"{replace_invalid_char(row['orig Songs Names'])}_{original_start_time}_original.png"
    file_path = os.path.join(start_time_folder_path, file_name)
    if not os.path.isfile(file_path):
        if DIAGRAM == 'mel':
            # Save the mel-spectrogram as a PNG image
            save_mel_spectrogram(audio_clip_origin, file_path)
            if BPM_DURATION: 
                for semitone in SEMITONE_LS:
                    audio_origin_shifted = shift_audio(audio_clip_origin, SAMPLE_RATE, semitone)
                    file_name = f"{replace_invalid_char(row['orig Songs Names'])}_{original_start_time}_original_{semitone}.png"
                    file_path = os.path.join(start_time_folder_path, file_name)
                    save_mel_spectrogram(audio_origin_shifted, file_path)
        elif DIAGRAM == 'chroma':
            save_chroma_feature(audio_clip_origin, file_path)
            if BPM_DURATION: 
                for semitone in SEMITONE_LS:
                    audio_origin_shifted = shift_audio(audio_clip_origin, SAMPLE_RATE, semitone)
                    file_name = f"{replace_invalid_char(row['orig Songs Names'])}_{original_start_time}_original_{semitone}.png"
                    file_path = os.path.join(start_time_folder_path, file_name)
                    save_chroma_feature(audio_origin_shifted, file_path)


# Read the Excel file
df = pd.read_csv(EXCEL_PATH)
numrows = df.shape[0]
# Main processing loop
for index, row in df.iterrows():
    process_audio_files(row)
    print(f"done {index} out of {numrows}")
    # input()


print("Dataset creation is complete.")
