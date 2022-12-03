#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 December 03, 22:49:19
@last modified : 2022 December 04, 00:41:53
"""

import gradio as gr
from summarizer import DefaultMapping, load_model, speech_to_summary_conditional

css = open("style.css", "r").read()

class mapping(DefaultMapping):
    encoder = 'cuda:0'
    decoder = 'cuda:0'

model = load_model(mapping=mapping)

def fn(audio_fname):
    summary = speech_to_summary_conditional(model, audio_fname, 200, mapping=mapping)
    return summary

demo = gr.Interface(fn=fn,
        inputs=[
            gr.Audio(type="filepath")
            ],
        outputs="text")

demo.launch(server_name='0.0.0.0')
