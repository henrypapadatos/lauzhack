#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 December 03, 22:49:19
@last modified : 2022 December 04, 03:22:02
"""

import gradio as gr
from summarizer import DefaultMapping, load_model, speech_to_summary_conditional, speech_to_text, text_to_summary
from youtube import download_from_youtube

css = open("style.css", "r").read()

class mapping(DefaultMapping):
    encoder = 'cuda:0'
    decoder = 'cuda:0'

model = load_model(mapping=mapping)

def transcribe(audio, mic, yt):
    if audio is not None:
        txt, _ = speech_to_text(model, audio, mapping=mapping)
    elif mic is not None:
        txt, _ = speech_to_text(model, mic, mapping=mapping)
    elif yt is not None:
        fname = transcribe_from_yt(yt)
        txt, _ = speech_to_text(model, fname, mapping=mapping)
    return txt

def transcribe_from_yt(url):
    fname = download_from_youtube(url)
    summary = speech_to_summary_conditional(model, fname, 200, mapping=mapping)
    return summary

def summarize(transcription):
    summary = text_to_summary(transcription)
    return summary

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tab("Audio File"):
                audio = gr.Audio(type="filepath", label="Audio File")
            with gr.Tab("Microphone"):
                mic = gr.Audio(source="microphone", type="filepath", label="Microphone")
            with gr.Tab("YouTube"):
                youtube = gr.Textbox(label="YouTube URL", lines=1, placeholder="https://www.youtube.com/watch?v=f_NoW_npKMA&ab_channel=HenryPapadatos")
            transcribe_btn = gr.Button("Transcribe")
            with gr.Accordion("Transcription", collapsed=True):
                transcription_txt = gr.Textbox(label=None)
        with gr.Column():
            with gr.Tab("Summary"):
                summary_txt = gr.Textbox(label=None)
                summary_btn = gr.Button("Summarize")
    transcribe_btn.click(transcribe, [audio, mic, youtube], transcription_txt)
    summary_btn.click(summarize, transcription_txt, summary_txt)
demo.launch(server_name="0.0.0.0")
