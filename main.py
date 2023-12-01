import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps    

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
torch.cuda.empty_cache()
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from xception import *
from xception import load_model
from triplet_loss import TripletLoss
from network_dataset import NetworkDataset
from utils import *
from make_data import *



# Hyperparameters
# sample rate of songs
SAMPLE_RATE = 48000


def main():

	#
	if MODE == 'train':

		# Load the training dataset
		folder_dataset = datasets.ImageFolder(root=DATASET_PATH)

		# Resize the images and transform to tensors
		transformation = transforms.Compose([transforms.Resize((299,299)),
											 transforms.ToTensor()
											])

		# Initialize the network
		dataset = NetworkDataset(imageFolderDataset=folder_dataset,
												transform=transformation)

		# Load the training dataset
		train_dataloader = DataLoader(dataset,
								shuffle=True,
								num_workers=8,
								batch_size=16)

		# Initialize network
		net = xception(pretrained = True, to_cuda = CUDA, reload_previous = RELOAD_PREVIOUS, RELOAD_PATH = RELOAD_PATH)
		# Define the loss metric as Triplet Loss
		criterion = TripletLoss()
		# Define the optimizer used as Adam
		optimizer = optim.Adam(net.parameters(), lr = 0.001 )


		# Record loss value during training
		counter = []
		loss_history = []
		loss_epoch_history = np.zeros([TRAINING_EPOCH, 2])
		# If training is restarted from a savepoint, load previous loss history
		if RELOAD_PREVIOUS:
			data = np.loadtxt("{0}_loss.csv".format(MODEL_PATH), delimiter=',')
			nonzerorow, nonzerocol = np.nonzero(data)
			cutrow = np.min([np.max(nonzerorow), STARTING_EPOCH])
			nonzerodata = data[:cutrow+1,:]
			loss_epoch_history = np.vstack([nonzerodata, loss_epoch_history])
		iteration_number= 0
		# Ending index of epoch
		ending_epoch = STARTING_EPOCH+TRAINING_EPOCH
		# Iterate throught the epochs
		for epoch in range(STARTING_EPOCH, ending_epoch):
			# Initialize loss
			epoch_loss_ave = 0.0
			# Iterate over batches
			for i, (img0, img1, img2) in enumerate(train_dataloader, 0):
				if CUDA:
					# Send the images and labels to CUDA
					img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()

				# Zero the gradients
				optimizer.zero_grad()

				# Pass in the two images into the network and obtain two outputs
				output1, output2, output3 = net(img0, img1, img2)

				# Pass the outputs of the networks and label into the loss function
				loss_triplet = criterion(output1, output2, output3)

				# Calculate the backpropagation
				loss_triplet.backward()

				# Optimize
				optimizer.step()

				# Accumulate loss value
				epoch_loss_ave += loss_triplet.item()

				# Every 10 batches print out the loss
				if i % 10 == 0 :
					print(f"Epoch number {epoch} batch {i}\n Current loss {loss_triplet.item()}\n")
					iteration_number += 10
					counter.append(iteration_number)
					loss_history.append(loss_triplet.item())
			# Average loss value over batches
			epoch_loss_ave = epoch_loss_ave/i
			print(f"Epoch number {epoch}\n Current loss {epoch_loss_ave}\n")
			# Record loss value
			loss_epoch_history[epoch,0] = epoch
			loss_epoch_history[epoch,1] = epoch_loss_ave
			# Save current savepoint
			torch.save(net.state_dict(), "{0}_{1}.pth".format(MODEL_PATH, epoch))
			# Save current loss value
			np.savetxt("{0}_loss.csv".format(MODEL_PATH), loss_epoch_history, delimiter=',')

	elif MODE == 'mean_distance':

		# Load the training dataset
		folder_dataset = datasets.ImageFolder(root=DATASET_PATH)

		# Resize the images and transform to tensors
		transformation = transforms.Compose([transforms.Resize((299,299)),
											 transforms.ToTensor()
											])

		 # Initialize the network
		dataset = NetworkDataset(imageFolderDataset=folder_dataset,
												transform=transformation)

		# Load the training dataset
		validate_dataloader = DataLoader(dataset,
								shuffle=False,
								num_workers=8,
								batch_size=128)

		# Initialize the network
		net = xception(pretrained = True, to_cuda = CUDA, reload_previous = RELOAD_PREVIOUS, RELOAD_PATH = RELOAD_PATH)
		# Define evaluation metric as mean distance
		criterion = MeanDistance()

		# Record loss value during training
		ending_epoch = STARTING_EPOCH+TRAINING_EPOCH
		# Initialize array to record distance history
		distance_history = np.zeros([ending_epoch,3])
		# Iterate throught the epochs
		for epoch in range(STARTING_EPOCH, ending_epoch):
			# Load savepoint at each epoch
			LOAD_PATH = "{1}_{0}.pth".format(epoch, MODEL_PATH)
			load_model(net, path = LOAD_PATH)
			# Initialize mean distance
			epoch_distance_n = 0.0
			epoch_distance_p = 0.0
			# Iterate over batches
			for i, (img0, img1, img2) in enumerate(validate_dataloader, 0):

				if CUDA:
					# Send the images and labels to CUDA
					img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()
				with torch.no_grad():
					# Pass in the two images into the network and obtain two outputs
					output1, output2, output3 = net(img0, img1, img2)

				# Pass the outputs of the networks and label into the loss function
				mean_distance_p, mean_distance_n = criterion(output1, output2, output3)
				# Accumulate mean distance
				epoch_distance_n += mean_distance_n.item()
				epoch_distance_p += mean_distance_p.item()
			# Average distance over batches
			epoch_distance_n = epoch_distance_n/i
			epoch_distance_p = epoch_distance_p/i
			print(f"Epoch number {epoch}\n Mean positive distance {epoch_distance_p}\n Mean negative distance {epoch_distance_n}\n")
			# Record the mean distances
			distance_history[epoch,1] = epoch_distance_p
			distance_history[epoch,2] = epoch_distance_n
			distance_history[epoch,0] = epoch
			np.savetxt("{0}_distance.csv".format(MODEL_PATH), distance_history, delimiter=',')

	elif MODE == 'validate':
		# Load the training dataset
		folder_dataset = datasets.ImageFolder(root=VALIDATE_PATH)

		# Resize the images and transform to tensors
		transformation = transforms.Compose([transforms.Resize((299,299)),
											 transforms.ToTensor()
											])

		# Initialize the network
		dataset = NetworkDataset(imageFolderDataset=folder_dataset,
												transform=transformation)

		# Load the training dataset
		validate_dataloader = DataLoader(dataset,
								shuffle=False,
								num_workers=8,
								batch_size=128)
		# Initialize the network
		net = xception(pretrained = True, to_cuda = CUDA, reload_previous = RELOAD_PREVIOUS, RELOAD_PATH = RELOAD_PATH)
		# Define the loss metric as Triplet Loss
		criterion = TripletLoss()

		ending_epoch = STARTING_EPOCH+TRAINING_EPOCH

		validate_history = np.zeros([ending_epoch,2])
		# Iterate throught the epochs
		for epoch in range(STARTING_EPOCH, ending_epoch):
			LOAD_PATH = "{1}_{0}.pth".format(epoch, MODEL_PATH)
			load_model(net, path = LOAD_PATH)
			epoch_loss_ave = 0.0
			# Iterate over batches
			for i, (img0, img1, img2) in enumerate(validate_dataloader, 0):
				# def closure():
					# loss = criterion(output1, output2, output3)
					# loss.backward()
					# return loss
				if CUDA:
					# Send the images and labels to CUDA
					img0, img1, img2 = img0.cuda(), img1.cuda(), img2.cuda()
				with torch.no_grad():
					# Pass in the two images into the network and obtain two outputs
					output1, output2, output3 = net(img0, img1, img2)

				# Pass the outputs of the networks and label into the loss function
				loss_triplet = criterion(output1, output2, output3)

				epoch_loss_ave += loss_triplet.item()
			epoch_loss_ave = epoch_loss_ave/i
			print(f"Epoch number {epoch}\n Validation loss {epoch_loss_ave}\n")
			validate_history[epoch,1] = epoch_loss_ave
			validate_history[epoch,0] = epoch
			np.savetxt("{0}_loss_val.csv".format(MODEL_PATH), validate_history, delimiter=',')

	elif MODE == 'test':

		# Initialize the network
		net = xception(pretrained = True, to_cuda = CUDA, reload_previous = RELOAD_PREVIOUS, RELOAD_PATH = RELOAD_PATH)
		# Resize the images and transform to tensors
		transformation = transforms.Compose([transforms.Resize((299,299)),
											 transforms.ToTensor()
											])
		# Define model used
		if 'Melspectrogram' in MODEL_PATH:
			mel = True
			inputname = 'melspectrogram'
		elif 'Chroma' in MODEL_PATH:
			mel = False
			inputname = 'chroma'
		if '10s' in MODEL_PATH:
			seg10s = True
			segname = '10s'
		elif 'preprocess' in MODEL_PATH:
			seg10s = False
			segname = 'preprocessed'

		# Grab audios
		file_list = os.listdir(TEST_PATH)

		# Initialize lists
		song_list = []
		duration_list = []
		seg_list = []
		num_seg_list = []
		feature_list = []

		# Define batch size to run at once
		run_batch = 48

		for i in range(len(file_list)):
			# Find all the mp3 files in the TEST_PATH folder
			if 'mp3' in file_list[i]:
				filename = file_list[i].split('.')[0]
				# Get the song name
				song_list.append(filename)
				# Load the song
				y, sr = librosa.load(os.path.join(TEST_PATH, file_list[i]), sr=SAMPLE_RATE)
				# Select the slicing length
				if seg10s:
					seg = 10
				else:
					seg = calculate_eightbars_duration(y, SAMPLE_RATE)
				seg_list.append(seg)

				# Get the duration of the song
				duration = librosa.get_duration(y=y, sr=SAMPLE_RATE)
				duration_list.append(duration)

				# Compute how many segments could be sliced out
				numseg = num_segment(duration, seg)
				num_seg_list.append(numseg)

				# Initialize tensor to save transformed diagrams and outputted feature maps
				segarrs = torch.empty((int(numseg), 3, 299, 299))
				featuremap = torch.empty((int(numseg), 1000))

				# Iterate over the segments
				for j in range(int(numseg)):
					# Define the slice starting and ending time
					start_time, end_time = seg_interval(j, duration, seg)
					start_sample = librosa.time_to_samples(start_time, sr=SAMPLE_RATE)
					end_sample = librosa.time_to_samples(end_time, sr=SAMPLE_RATE)
					# Slice the audio
					audio_clip = y[start_sample:end_sample]
					# Generate appropraite diagrams
					if mel:
						img = get_mel_spectrogram(audio_clip)
					else:
						img = get_chroma_feature(audio_clip)

					# Transform the diagrams and save them
					arr = transformation(img)
					segarrs[j,:,:,:] = arr

					# Compute feature maps over batches
					if (j%run_batch == 0 and j!=0):
						batch_ind = j//run_batch
						# Compute batch starting and ending index in saved input array
						start_ind = (batch_ind-1)*run_batch
						end_ind = np.min([(batch_ind)*run_batch,int(numseg)])
						# Select the batch inputs
						segbatch = segarrs[start_ind:end_ind,:,:,:]
						if CUDA:
							segbatch = segbatch.cuda()
						with torch.no_grad():
							# Compute the feature map
							featuremap_batch = net.forward_once(segbatch)
						# Save the feature map
						featuremap[start_ind:end_ind,:] = featuremap_batch.cpu()

					if (j==int(numseg)-1):
						batch_ind = j//run_batch
						# Compute batch starting and ending index in saved input array
						start_ind = (batch_ind)*run_batch
						end_ind = int(numseg)
						# Select the batch inputs
						segbatch = segarrs[start_ind:end_ind,:,:,:]
						if CUDA:
							segbatch = segbatch.cuda()
						with torch.no_grad():
							# Compute the feature map
							featuremap_batch = net.forward_once(segbatch)
						# Save the feature map
						featuremap[start_ind:end_ind,:] = featuremap_batch.cpu()
				# Save the computed feature maps
				feature_list.append(featuremap)

		# Initialize array to save similarity scores
		record = np.zeros([int(num_seg_list[0]*num_seg_list[1]),8])
		counter = 0
		# Get the duration, segment length and computed feature maps of the pair
		duration1 = duration_list[0]
		duration2 = duration_list[1]
		seg1 = seg_list[0]
		seg2 = seg_list[1]
		featuremap1 = feature_list[0]
		featuremap2 = feature_list[1]
		# Iterate over feature maps of all segments
		for i in range(int(num_seg_list[0])):
			out1 = featuremap1[[i],:]
			# Compute the corresponding segment starting and ending time
			start_time1, end_time1 = seg_interval(i, duration1, seg1)
			for j in range(int(num_seg_list[1])):
				out2 = featuremap2[[j],:]
				# Compute the corresponding segment starting and ending time
				start_time2, end_time2 = seg_interval(j, duration2, seg2)
				# Compute similarity scores of pairs
				[w_score, edist, cos_sim, p_corr] = weighted_score(out1, out2)
				# Record the computed results
				record[counter, 0] = start_time1
				record[counter, 1] = end_time1
				record[counter, 2] = start_time2
				record[counter, 3] = end_time2
				record[counter, 4] = w_score
				record[counter, 5] = edist
				record[counter, 6] = cos_sim
				record[counter, 7] = p_corr
				counter += 1

		# Find the minimum euclidean distance and corresponding index
		min_score = np.min(record[:,5])
		min_index = np.argmin(record[:,5])

		print("For Case {9}, using {0} and {1} segments, the min euclidean distance is {2}. \n Occurs at {3} ({4}:{5}) and {6} ({7}:{8}). \n".format(inputname,segname, min_score, song_list[0], record[min_index,0], record[min_index,1], song_list[1], record[min_index,2], record[min_index,3],TEST_PATH.split(' ')[-1]))
		# Save records
		recordname = "{0}_{1}_{2}_{3}".format(inputname, segname, song_list[0], song_list[1])
		np.savetxt("{1}/{0}.csv".format(recordname,TEST_PATH),np.array(record), delimiter=',')

# paths of the best savepoints for the four models we have trained at your disposal 
best_savepoints = [
	"Best savepoints/model_chromafeature10s_136.pth",
	"Best savepoints/model_chroma_preprocessed_145.pth",
	"Best savepoints/model_spectrogram10s_147.pth",
	"Best savepoints/model_spectrogram_preprocessed_147.pth"
]

# There are four different modes to choose from:
#   'train': start training the model using Adam optimizer and Triplet loss.
# 		If RELOAD_PREVIOUS is set to False, starting_epoch will be set to 0, 
#   and the model will be trained from scratch.
# 		If the user wants to start training from a saved savepoint, set 
#   RELOAD_PREVIOUS to True and there will be a prompt asking for the starting 
#   epoch of the training. The user should input the epoch corresponding to
#   the savepoint.          
# 	'validate': the script will reload savepoints in MODEL_PATH and compute the loss 
#  value of the validation dataset for each savepoint. 
#   'mean_distance': the script will reload savepoints in MODEL_PATH and compute the mean distance 
#  value of the training dataset for each savepoint. 
#   'test': the script to load the pair of mp3 files in TEST_PATH. By reloading the savepoint 
#  in RELOAD_PATH, the model will slice segments of the two mp3 files in TEST_PATH based on 
# the model_name and compute the feature maps of the segments. A minimum Euclidean distance 
# between feature maps and the corresponding segments are printed to display.
MODE = 'train'
if MODE == 'test' or MODE == 'validate' or MODE == 'mean_distance':
	RELOAD_PREVIOUS = True 
else:
	response = input("Do you want to reload a savepoint for retraining? (Yes or No) ")
	RELOAD_PREVIOUS = True if response == 'Yes' else False
# path pointing to the training dataset
DATASET_PATH = "./chroma_feature10s_dataset/training"
# path pointing to the validation dataset
VALIDATE_PATH = "./chroma_feature10s_dataset/testing"
# path pointing to the folder containing a pair of mp3 files to be compared
# the folder should ONLY contain two mp3 files
TEST_PATH = "./Audio/Case 1"
# path pointing to the folder for saving training savepoints
MODEL_PATH = "./chroma_feature10s_savepoint"
# name for saving your model
model_name = "model_chroma10s"
# controls the script whether or not to use GPU resources. 
CUDA = True
# epoch duration of the training 
TRAINING_EPOCH = 150
if RELOAD_PREVIOUS:
	if MODE == 'train':
		STARTING_EPOCH = int(input("Enter the starting epoch for retraining: "))
		RELOAD_PATH = "{1}/{2}_{0}.pth".format(STARTING_EPOCH, MODEL_PATH, model_name)
	else:
		STARTING_EPOCH = 0
		RELOAD_PATH = input("Enter the path to the savepoint: ")
else:
	STARTING_EPOCH = 0
	RELOAD_PATH = ""


if __name__ == '__main__':
	main()
