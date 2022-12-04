import os
import pafy 

def download_from_youtube(url = "https://www.youtube.com/watch?v=f_NoW_npKMA&ab_channel=HenryPapadatos"):
        
    video = pafy.new(url)
    
    bestaudio = video.getbestaudio()
    path = bestaudio.download()

    return path 