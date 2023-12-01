import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import pandas as pd
import torch.nn.functional as F


# ------------ string related
def convert_time_to_sec(time_str):
    # convert time string like ['1:27'] to seconds
    if "," in time_str:
        return [convert_time_to_sec(i) for i in time_str.split(",")]

    # if "[" in time_str:
    time_str = time_str.replace("[", "")
    time_str = time_str.replace("]", "")
    time_str = time_str.replace("'", "")

    if ":" in time_str:
        time_str = time_str.split(":")
    elif "_" in time_str:
        time_str = time_str.split("_")
    return int(time_str[0]) * 60 + int(time_str[1])

def replace_invalid_char(string):
	invalid_char = ['/',':','*','?','"','<','>','|']
	new_string = ''
	for s in string:
		if s in invalid_char:
			s = '_'
		new_string += s
	return new_string

def remove_special_char(string):
	test_str = ''.join(letter for letter in string if letter.isalnum())
	return test_str
#--------------------------------------------


# ------------------------- audio related
def stretch_audio(audio, sr, rate):
	new_audio = librosa.effects.time_stretch(audio, rate=rate)
	return new_audio

def shift_audio(audio, sr, semitone):
	y_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitone)
	return y_shifted

def bpm_estimation(audio, sr):
	bpm, _ = librosa.beat.beat_track(y=audio, sr=sr)
	return bpm

def calculate_eightbars_duration(audio, sr):
	bpm = bpm_estimation(audio, sr)
	# assuming a 44 time signature
	secs = 60/bpm*4*8
	return secs
# --------------------------------------

# ---------------------for dataset conversion
def list_of_existing_sampletime(dir_list):
    return np.array(dir_list, dtype = np.int32)

def find_closest_number(array, target):
  # Find the minimum distance between the target value and each number in the array.
  distances = [np.abs(target - number) for number in array]
  # Return the index of the number with the minimum distance.
  return distances.index(min(distances))

def save_mel_spectrogram(audio_clip, file_path):
    # Produce the mel-spectrogram
    S = librosa.feature.melspectrogram(y=audio_clip, sr=SAMPLE_RATE)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Save the mel-spectrogram
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(S_DB, sr=SAMPLE_RATE)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.close()

def save_chroma_feature(audio_clip, file_path):
     # Produce the chroma feature
    S = np.abs(librosa.stft(audio_clip, n_fft=4096))**2
    S = librosa.feature.chroma_stft(y=audio_clip, sr=SAMPLE_RATE)
    chroma = librosa.amplitude_to_db(S, ref=np.max)

    # Save the chroma feature
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(chroma, sr=SAMPLE_RATE)
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', pad_inches = 0, transparent = True)
    plt.close()

def get_mel_spectrogram(audio_clip):
    S = librosa.feature.melspectrogram(y=audio_clip, sr=SAMPLE_RATE)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 10))
    librosa.display.specshow(S_DB, sr=SAMPLE_RATE)
    plt.tight_layout()
    tempname = "/content/drive/MyDrive/Music_Plagiarism/temp_{0}.png".format(time.time())
    plt.savefig(tempname, bbox_inches='tight', pad_inches = 0, transparent = True)
    img = Image.open(tempname)
    img = img.convert("RGB")
    os.remove(tempname)
    plt.close()
    return img

def get_chroma_feature(audio_clip):
    S = np.abs(librosa.stft(audio_clip, n_fft=4096))**2
    S = librosa.feature.chroma_stft(y=audio_clip, sr=SAMPLE_RATE)
    chroma = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 10))
    librosa.display.specshow(chroma, sr=SAMPLE_RATE)
    plt.tight_layout()
    tempname = "./temp_{0}.png".format(time.time())
    plt.savefig(tempname, bbox_inches='tight', pad_inches = 0, transparent = True)
    img = Image.open(tempname)
    img = img.convert("RGB")
    img.show()
    os.remove(tempname)
    plt.close()
    return img
#--------------------------------------------

# ------------------------ for images
# Showing images
def imshow(img, text=None):
	npimg = img.numpy()
	plt.axis("off")
	if text:
		plt.text(75, 8, text, style='italic',fontweight='bold',
			bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})

	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

# Plotting data
def show_plot(iteration,loss):
	plt.plot(iteration,loss)
	plt.show()

#-------------------------------------------

# -------------- similarity score functions
def cos_sim_score(output1, output2):
	score = F.cosine_similarity(output1,output2, dim = 1)
	return score

def pearson_corr_score(output1, output2):
	xmean = torch.mean(output1)
	ymean = torch.mean(output2)
	p_score = torch.sum((output1-xmean)*(output2-ymean))/torch.sqrt(torch.sum((output1-xmean)**2)*torch.sum((output2-ymean)**2))
	return p_score

def weighted_score(output1,output2):
	edist = torch.dist(output1, output2)**2
	cos_sim = cos_sim_score(output1, output2)
	p_corr = pearson_corr_score(output1, output2)
	w_score = 0.2*edist+0.4*cos_sim+0.4*p_corr
	return [w_score, edist, cos_sim, p_corr]

def num_segment(duration, seg):
	interval = seg/2
	num = np.ceil((duration-seg)/interval)
	return num

def seg_interval(ind, duration, seg):
	interval = seg/2
	start_time = ind*interval
	end_time = start_time+seg
	if end_time>duration:
		end_time = duration
	return start_time, end_time
# ---------------------------------------