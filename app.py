#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 December 03, 22:49:19
@last modified : 2022 December 04, 05:56:31
"""

import gradio as gr
from summarizer import DefaultMapping, load_model, speech_to_summary_conditional, speech_to_text, text_to_summary
from youtube import download_from_youtube
from conversation import Conversation

css = open("style.css", "r").read()

class mapping(DefaultMapping):
    encoder = 'cpu'
    decoder = 'cpu'

model = load_model(mapping=mapping)
conv_bot : Conversation = None

def transcribe(audio, mic, yt):
    if audio is not None:
        txt, language = speech_to_text(model, audio, mapping=mapping)
    elif mic is not None:
        txt, language = speech_to_text(model, mic, mapping=mapping)
    elif yt is not None:
        fname = transcribe_from_yt(yt)
        txt, language = speech_to_text(model, fname, mapping=mapping)
    return txt, language

def transcribe_from_yt(url):
    fname = download_from_youtube(url)
    summary = speech_to_summary_conditional(model, fname, 200, mapping=mapping)
    return summary

def summarize(transcription, language):
    summary = text_to_summary(transcription, language)
    return summary

def chat(message, history, transcription, language):
    global conv_bot
    global ANSWERING_QUESTION
    ANSWERING_QUESTION = ANSWERING_QUESTION or False
    history = history or []
    language = language or 'en'

    conv_bot = conv_bot or Conversation(transcription, language)
    if conv_bot.explanations != transcription:
        conv_bot = Conversation(transcription)
        history = []

    if ANSWERING_QUESTION:
        question = history[-1][1]
        response = conv_bot.evalutate_answer(question,message)
    else:
        response = conv_bot.ask_question(message)
    history.append((message, response))
    ANSWERING_QUESTION = False
    return history, history


def ask_a_question(history, transcription, language):
    global conv_bot
    global ANSWERING_QUESTION
    history = history or []
    language = language or 'en'

    conv_bot = conv_bot or Conversation(transcription, language)
    if conv_bot.explanations != transcription:
        conv_bot = Conversation(transcription,language)
        history = []

    question = conv_bot.evaluate_understanding()
    history.append(("Ask me a question", question))
    ANSWERING_QUESTION = True
    return history, history


with gr.Blocks() as demo:
    language_state = gr.State()
    gr.HTML(
            """
            <div style="text-align: center;">
              <div
                style="
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  Patrick
                </h1>
              </div>
              <div
                style="
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h2 id='subtitle'>A powerful easy to use tool that can summarize and interact based on an audio context</h2>
                </div>

              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
            </div>
        """
        )
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
                transcription_txt = gr.Markdown(label=None)
        with gr.Column():
            with gr.Tab("Summary"):
                summary_txt = gr.Textbox(label=None)
                summary_btn = gr.Button("Summarize")
            with gr.Tab("Ask questions"):
                message = gr.Textbox(label="Message", lines=1, placeholder="What is the weather like today?")
                submit = gr.Button("Submit")
                chat_state = gr.State()
                chatbot = gr.Chatbot().style(color_map=("#bd3a8a", "#1069db"))
                submit.click(chat, [message, chat_state, transcription_txt, language_state], [chatbot, chat_state])
                with gr.Row():
                    ask = gr.Button("Ask me a question")
                    ask.click(ask_a_question, [chat_state, transcription_txt, language_state], [chatbot, chat_state])
                    


    transcribe_btn.click(transcribe, [audio, mic, youtube], [transcription_txt, language_state])
    summary_btn.click(summarize, [transcription_txt, language_state], summary_txt)

demo.launch()
