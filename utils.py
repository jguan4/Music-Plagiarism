from pydub import AudioSegment 
import librosa
import numpy as np
import matplotlib.pyplot as plt


def load_mp3(audio_path):
	song = AudioSegment.from_file(audio_path, format="mp3")
	return song

def trim_mp3(mp3_audio, start_time, duration_sec=10):
	song = AudioSegment.from_mp3(mp3_audio) 

	# pydub does things in milliseconds 
	seconds = duration_sec * 1000
	cut_song = song[start_time*1000:seconds] 
	  
	return  cut_song

def convert_mp3_to_melspectrogram(audio):
	y, sr = librosa.load(audio, sr = None)
	S = librosa.feature.melspectrogram(y=y, sr=sr)
	S_dB = librosa.power_to_db(S, ref=np.max)
	fig, ax = plt.subplots()
	img = librosa.display.specshow(S_dB, x_axis='time',
							 y_axis='mel', sr=sr,
							 fmax=8000, ax=ax)
	return img

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
