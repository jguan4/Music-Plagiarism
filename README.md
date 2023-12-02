# Detect Music Plagiarism using Siamese CNN
Music plagiarism has been a legally gray area for many decades. The complexity of music making contributes to the difficulty of defining plagiarism among songs. In the past, music plagiarism occurred in two contexts: with a musical idea (melody, motif) or sampling (taking a portion of one sound and reusing it in a different song). According to U.S. copyright law, in the absence of a confession, musicians who accuse others of stealing their work must prove "access"—the alleged plagiarizer must have heard the song—and "similarity"—the songs must share unique musical components. The focus of this project will be on the similarity between two songs, i.e. we will train a network as a similarity metric between two segments of audios. Due to the limitation of time and resources, this project will purely focus on audio similarity. Similarity in lyrics, for example, will be out of the scope of this project. The model trained as a similarity predictor for pairs of songs could offer insights into how similar two songs are. Having clear criteria for plagiarism will both protect the original artists from being exploited, and will also allow the free-spirited artists to innovatively create new musical pieces. The judiciary system and the  whole music industry, including artists, record labels and tech companies (e.g., Spotify, Deezer, Tidal) would benefit from such a model. 

## Data Collection
Our approach follows the one used in [[1]](#1) and expand on it. First, we need to gather our dataset. The dataset we used for this project consists of songs appearing in WhoSampled.com. WhoSampled.com provides lists of songs that have sampled another song, along with timestamps of where the relevant sample begins. We start by identifying pairs of songs from whosampled.com, along with the relevant timestamps. The pairs are then downloaded from YouTube as .mp3 files (for a total of 1128 pairs of songs). These tasks are accomplished through ```WhoSampledWebScrapper.py```, which generates a csv file (WSdata3.csv) that contains song titles and sampled segments time stamps. Then ```download_fromExcel.py``` download the songs from YouTube. 

With all the audio files downloaded, we considered two different methods of extracting the sampled portion. The first is slicing ten seconds of the sampled-segments. The second is slicing the duration equivalent to 8 bars of the sampled-segments, in attempts to generate diagrams invariant under different tempo. 

We considered two image-representation of music: mel-spectrogram and chroma features. For mel-spectrograms, fast Fourier transform is first applied on audio signals and then frequencies are converted to the mel scale, which is a unit of pitch such that equal distances in pitch sounded equally distant to the listener. These mel-scaled frequencies are visualized in terms of their amplitudes to create mel-spectrogram. For chroma features, frequencies are binned into twelve different pitch classes. Chroma features are known for capturing harmonic and melodic characteristics of music, while being robust to changes in timbre and instrumentation. 

We generated datasets using ```make_data.py```. The created datasets are listed below:
links to existing dataset 
- [Chroma Feature 10s Dataset](https://drive.google.com/drive/folders/18AKPPXjNuXl5v4-dVjbhUdag32l4ZSdQ?usp=drive_link)
- [Chroma Feature 8-bar Dataset](https://drive.google.com/drive/folders/1--NH_NAi_zO0-BtGEvVQAcNun_7fgyYX?usp=drive_link)
- [Mel-Spectrogram 10s Dataset](https://drive.google.com/drive/folders/1-jA_69VR10KpUnnCy2_rgPSBkXwaOgw5?usp=drive_link)
- [Mel-Spectrogram 8-bar Dataset](https://drive.google.com/drive/folders/1el-5bOel0v4ib6v_JyldbLSXtuXc-EKn?usp=sharing)

We separated the collected data into training and validation datasets. 

## Approach
For our approach, we followed [[1]](#1). Image datasets representing music segments were inputted through a Siamese Convolutional Neural Networks (CNN). The outputs of the CNN were vectors of length 1000, called feature maps. The CNN used here was Xception [[2]](#2). Xception was known for better generability in classification. 
The Euclidean distances between feature maps could represent the similarity between two inputs. Ideally, two song segments with similar musical features inputted into the network should output feature maps that are close to each other in Euclidean distance, while two segments that are very different should have a large distance between their feature maps. To achieve this objective, Triplet Loss was used during training. The triplet loss is defined as 

$L(A,P,N) = \max(D(A,P)-D(A,N)+M, 0)$

where $A$ is the anchor image randomly selected from the dataset, $P$ is a positive image with the same label and $N$ is a negative image with a different label. Before training starts, we preload Xception weights and biases trained on ImageNet dataset. We train the network for 150 epochs using Adam optimizer. 

## Run Notebook
Our training was done on Google Colab using the ```Music_Plagiarism.ipynb``` notebook. 

## Result
Our four models were able to separate the different songs with a large margin of separation on the training dataset. 

![Model trained on Chroma Feature 10s dataset evaluated mean distance between similar and dissimilar songs during training](https://github.com/jguan4/Music-Plagiarism/assets/28985094/90c518f7-9985-482c-be85-0cf1dd7aa0bc) 

Fig1: Model trained on Chroma Feature 10s dataset evaluated mean distance between similar and dissimilar songs during training

![model_chroma_preprocessed_mean_distance](https://github.com/jguan4/Music-Plagiarism/assets/28985094/00d3623d-c6e9-4e21-9126-3af474041f48)

Fig2: Model trained on Chroma Feature 8-bar dataset evaluated mean distance between similar and dissimilar songs during training

![model_spectrogram10s_mean_distance](https://github.com/jguan4/Music-Plagiarism/assets/28985094/412324a5-56be-4c0f-a874-32b68905e55b)

Fig3: Model trained on Mel-Spectrogram 10s dataset evaluated mean distance between similar and dissimilar songs during training

![model_spectrogram_preprocessed_mean_distance](https://github.com/jguan4/Music-Plagiarism/assets/28985094/c3a7d8c0-f6bb-4e09-b7d0-df5fa4ebc572)

Fig4: Model trained on Mel-Spectrogram 8-bar dataset evaluated mean distance between similar and dissimilar songs during training

But the mean distances on validation dataset tell a different story. 
![model_chromafeature10s_mean_distance_validate](https://github.com/jguan4/Music-Plagiarism/assets/28985094/2a25dffc-2828-4e95-9358-6d17e93d74ef)

Fig5: Model trained on Chroma Feature 10s validation dataset evaluated mean distance between similar and dissimilar songs during training

![model_chroma_preprocessed_mean_distance_validate](https://github.com/jguan4/Music-Plagiarism/assets/28985094/414b2c9d-ff6d-4d15-b321-5b234adb54d4)

Fig6: Model trained on Chroma Feature 8-bar validation dataset evaluated mean distance between similar and dissimilar songs during training

![model_spectrogram10s_mean_distance_validate](https://github.com/jguan4/Music-Plagiarism/assets/28985094/ffe6e05a-f0bb-4ee0-96f7-d2159834f3fa)

Fig7: Model trained on Mel-Spectrogram 10s validation dataset evaluated mean distance between similar and dissimilar songs during training

![model_spectrogram_preprocessed_mean_distance_validate](https://github.com/jguan4/Music-Plagiarism/assets/28985094/bf46b1e5-3814-4de3-98fd-9cb861b6539d)

Fig8: Model trained on Mel-Spectrogram 8-bar validation dataset evaluated mean distance between similar and dissimilar songs during training

We observe mel-spectrogram 10 seconds samples is the dataset that has trained the best model, being able to separate the similar songs and dissimilar songs with a value around ten in the mean Euclidean distance. We used the best-performing 10-second mel-spectrogram model to calculate a similarity score for legal cases that we collected, 9 guilty, 6 settled, 6 acquitted. The similarity score is labeled as Euclidean Distance in the y-axis. 

![box_plot](https://github.com/jguan4/Music-Plagiarism/assets/28985094/1e0bd554-b79b-4044-a153-f44c76ce8444)

Fig9: Box plot of the Euclidean distance of courtcases. 

As we discussed before, the smaller the distance is, the more similar the two songs are. Our model produced very small distances for guilty cases, while larger distances for settled or acquitted cases. Even though the sample size for the court cases is small, and there is overlap between the boxplots of guilty and not-guilty cases, our model had a notable differentiation in distances between guilty and settled cases.

## Conclusion
In conclusion, our model was able to distinguish similar songs from dissimilar songs. Due to this, we believe that we can reasonably expect our model to give useful results when used by music industry professionals to assess whether a new song is similar enough to be a potential case of plagiarism. The use of Chroma features may have had poor results due to dominance of percussion used in our dataset. Chroma feature may prove more effective when used in cases where a melody or a tune is more saliently plagiarized. In the future, it might be beneficial to consider what type of plagiarism is being examined, such as beat vs melody, when selecting the dataset. Additionally, numeric representation of music might be preferable to image representation when processing the data. For example, using the values in the mel spectrogram rather than the image of it. Other avenues of possible improvement would in clyde investigate other types of networks. Possible architectures include Autoencoder, Long-Short Term Memory and Recurrent CNN.


## References
<a id="1">[1]</a> 
Kasif, G. and Thondilege, G. (2023). 
Exploring Music Similarity through Siamese CNNs using Triplet Loss on Music Samples. 
2023 International Research Conference on Smart Computing and Systems Engineering (SCSE).

<a id="2">[2]</a> 
Chollet, F. (2017). 
Xception: Deep learning with depthwise separable convolutions. 
Proceedings of the IEEE conference on computer vision and pattern recognition.

