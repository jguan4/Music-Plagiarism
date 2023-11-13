import os
import pandas as pd
from pytube import YouTube
from youtubesearchpython import VideosSearch
from moviepy.editor import *
from utils import *

# ------------------------------ for downloading songs from YouTube
def searchYoutube(search_text):
	videosSearch = VideosSearch(search_text, limit = 1)
	try:
		link = videosSearch.result()['result'][0]['link']
		return link
	except:
		print(search_text)
		return None

def downloadMP3_fromYouTube(filename, link):
	new_file = './YouTube_audios_updated/'+filename + '.mp3'
	yt = YouTube(link,use_oauth=True, allow_oauth_cache=True)
	try:
		video = yt.streams.filter(only_audio=True).first()
		if not os.path.isfile(new_file):
			out_file = video.download(output_path='./YouTube_audios_updated/')
			MP4ToMP3(out_file, new_file)
			# os.rename(out_file, new_file)
		return None
	except:
		print(filename)
		return filename

def MP4ToMP3(mp4, mp3):
	FILETOCONVERT = AudioFileClip(mp4)
	FILETOCONVERT.write_audiofile(mp3)
	FILETOCONVERT.close()

def download_fromExcel():
	df = pd.read_csv('WSdata3.csv')
	orig_song = df['orig Songs Names'].tolist()
	sampling_song = df['Sampling Song Names'].tolist()
	songs = orig_song+sampling_song

	songs = list(set(songs))
	failed_songs = []
	for i in range(len(songs)):
		songi = songs[i]
		link = searchYoutube(songi)
		if link:
			filename = downloadMP3_fromYouTube(replace_invalid_char(songi), link)
			if filename:
				failed_songs.append(songi)
		else:
			failed_songs.append(songi)
	df = pd.DataFrame({'failed songs':failed_songs})
	df.to_excel("failed_download_files.xlsx")
# ---------------------------------------------------------------