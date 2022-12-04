import os


def download_from_youtube(url = "https://www.youtube.com/watch?v=f_NoW_npKMA&ab_channel=HenryPapadatos"):
        
    fname = 'tmp.mp3'
    cmd_output = os.popen(f"yt-dlp -x --audio-format 'mp3' --output {fname} --prefer-ffmpeg {url}").read()

    return fname

