# -*- coding: utf-8 -*-
"""
Run this file to save song data from Whosampled.com as "WSdata.csv" file
Change the range of the first for loop to increase or decrease the number of songs scraped
"""

import pandas as pd
import matplotlib
import numpy
from time import sleep 
from bs4 import BeautifulSoup
import requests
import cloudscraper

class song:
    pass

#access intial webpage (list of most ampled tracks)
scraper = cloudscraper.create_scraper()
html=scraper.get("https://www.whosampled.com/most-sampled-tracks/")
wsSoup=BeautifulSoup(html.text,'html.parser')


#find song urls for each song listed on x many pages of the most sampled track list 

songURLs=[]
wsSoup.find('span', {'class':"trackInfo"}).a["href"]
songURLs.extend([x.a['href'] for x in wsSoup.find_all('span', {'class':"trackInfo"})])

for x in range(2,5):   #loops to different pages of the list of "Most Sampled Tracks" with 10 tracks per page, change the range to increase/decrease the number of original songs scraped
    html=scraper.get("https://www.whosampled.com/most-sampled-tracks/"+str(x)+"/")
    wsSoup=BeautifulSoup(html.text,'html.parser')
    wsSoup.find('span', {'class':"trackInfo"}).a["href"]
    songURLs.extend([x.a['href'] for x in wsSoup.find_all('span', {'class':"trackInfo"})])
    sleep(2)
    
        
origSongs=[]

for url in songURLs:
    #get name and artist of each song + go to the page for that song
    songHtml=scraper.get("https://www.whosampled.com"+url)          #wsSoup.find('span', {'class':"trackInfo"}).a["href"])
    songSoup=BeautifulSoup(songHtml.text,'html.parser')
    names=songSoup.h1.text.strip().split('\n\nby')
    origSong=song()
    origSong.name=names[0]+" ["+names[1]+"]"
    #origSong.artist=names[1]                #uncomment for seperate field for artist names
    origSong.sampledList=[]
    sleep(2)
    
    #go to "see all" page of sampling songs and get name and artist of each song on first page of list (15 songs)
   
    sampledHtml=scraper.get("https://www.whosampled.com"+url+'sampled/')
    sampledSoup=BeautifulSoup(sampledHtml.text,'html.parser')
    for track in sampledSoup.find_all("div", {"class":'trackDetails'}):
        sampledSong=song()

        sampledSong.name=track.a.text
        sampledSong.artist=track.find("span", {"class":'trackArtist'}).text.strip().split("(")[0].replace("by","")
        sampledSong.name=sampledSong.name + " [" +sampledSong.artist+ "]"
        sampledSongURL=track.a['href']
        
        
        #get sampled start times by going to page of sampling song
                                  
        sampledSongHtml=scraper.get("https://www.whosampled.com"+sampledSongURL)
        sampledSongSoup=BeautifulSoup(sampledSongHtml.text,'html.parser')
        sampledSong.sampleStart=sampledSongSoup.find("strong",{'id':"sample-dest-timing"}).text.strip().replace(","," ").replace("and"," ").split()
        sampledSong.origSampleStart=sampledSongSoup.find("strong",{'id':"sample-source-timing"}).text.strip().replace(","," ").replace("and"," ").split()
        
        origSong.sampledList.append(sampledSong)    #add info of sampling song to list attached to orginal song
        sleep(2)
             
    origSongs.append(origSong)
    
    
#set up columns for dataframe    
origSongsCol=[]
#origArtistCol=[]
sampledSongsNamesCol=[]
#sampledSongsArtistCol=[]
sampledStartTimeCol=[]
origSampleStartTimeCol=[]
for s in origSongs:
    origSongsCol.extend([s.name]*len(s.sampledList))
    #origArtistCol.extend([s.artist]*len(s.sampledList))  #uncomment and add to dataframe inputs to make a seperate column for artist names
    sampledSongsNamesCol.extend([x.name for x in s.sampledList])
    #sampledSongsArtistCol.extend([x.artist for x in s.sampledList])
    sampledStartTimeCol.extend([x.sampleStart for x in s.sampledList])
    origSampleStartTimeCol.extend([x.origSampleStart for x in s.sampledList])

#title, artist, songs that sampled, covers are put into a pandas dataframe then saved to "WSdata.csv" file
songDF=pd.DataFrame({'orig Songs Names':origSongsCol, "Sampling Song Names":sampledSongsNamesCol,  "Start of Sample":sampledStartTimeCol, "Start of sample in orig song":origSampleStartTimeCol})   #"orig artist names":origArtistCol, "sampled artist names":sampledSongsArtistCol, 
songDF.to_csv("WSdata.csv", header=False,index=False)
