import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


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
	w_score = 0.2*F.pairwise_distance(output1, output2, keepdim = True)**2+0.4*cos_sim_score(output1, output2)+0.4*pearson_corr_score(output1,output2)
	return w_score
#------------------------------------------



# from pydub import AudioSegment 
# def load_mp3(audio_path):
# 	song = AudioSegment.from_file(audio_path, format="mp3")
# 	return song

# def trim_mp3(mp3_audio, start_time, duration_sec=10):
# 	song = AudioSegment.from_mp3(mp3_audio) 

# 	# pydub does things in milliseconds 
# 	seconds = duration_sec * 1000
# 	cut_song = song[start_time*1000:seconds] 

# 	return  cut_song