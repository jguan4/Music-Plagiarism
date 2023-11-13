import mutagen
import os
import pandas as pd
from utils import *


# --------------- functions for checking old data
def check_incorrect_file():
	audio_folder = "./WSdata_audios/"
	files = os.listdir(audio_folder)
	wrong_file_titles = []
	wrong_file_artists = []
	wrong_audio_titles = []
	wrong_audio_artists = []
	for file in files:
		# check if it is an audio file
		extension = file[-3:]
		filename = file[:-4]
		# filename, extension = file.split('.')
		if extension == 'mp3':
			audio = mutagen.File(audio_folder+file)
			# check if string converted match
			title = audio["TIT2"].text[0]
			artist = audio["TPE1"].text[0]
			# proposed_filename = title + " [ "+artist+" ]"
			strind = filename.index('[')
			filename_title = filename[:strind-1]
			filename_artist = filename[strind+2:-2]

			# if it is not matching, save to excel file
			if (remove_special_char(filename_title) not in remove_special_char(title)) or (remove_special_char(filename_artist) not in remove_special_char(artist)):
				wrong_file_titles.append(filename_title)
				wrong_file_artists.append(filename_artist)
				wrong_audio_titles.append(title)
				wrong_audio_artists.append(artist)

	df = pd.DataFrame(data = {'file_title': wrong_file_titles,'file_artist': wrong_file_artists, 'title': wrong_audio_titles,'artist': wrong_audio_artists})
	df.to_excel("wrong_files.xlsx")
# ------------------------------------------------------



