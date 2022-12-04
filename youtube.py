import os
from pytube import YouTube

def download_from_youtube(url = "https://www.youtube.com/watch?v=f_NoW_npKMA&ab_channel=HenryPapadatos"):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    destination = '.'
    out_file = video.download(output_path=destination)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.wav'
    os.rename(out_file, new_file)
    print(yt.title + " has been successfully downloaded.")

    return new_file