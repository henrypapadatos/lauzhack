#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 December 03, 20:32:39
@last modified : 2022 December 04, 10:37:33
"""

import sys

sys.path.append('../whisper')

import os
import torch
import openai
import whisper
from dataclasses import dataclass
from whisper.decoding import DecodingOptions
from whisper.tokenizer import get_tokenizer

tasks = {
    "en": "Identify the key points in this message.", 
    "zh": "  从这条消息中提取主要思想。", 
    "de": "  Holen Sie die wichtigsten Ideen aus dieser Nachricht heraus.", 
    "es": "  Extraer las ideas principales de este mensaje.", 
    "ru": "  Извлеките основные идеи из этого сообщения.", 
    "ko": " 이 메시지에서 주요 아이디어를 추출하십시오.", 
    "fr": "  Extraire les idées principales de ce message.", 
    "ja": "  このメッセージから主要なアイデアを抽出してください。", 
    "pt": "  Extraia as ideias principais desta mensagem.", 
    "tr": "  Bu mesajdan ana fikirleri çikarin.", 
    "pl": "  Wyciągnij główne idee z tej wiadomości.", 
    "ca": "  Extreu les idees principals daquest missatge.", 
    "nl": "  Het hoofdidee uit dit bericht extraheren.", 
    "ar": " استخرج الأفكار الرئيسية من هذه الرسالة", 
    "sv": "  Extrahera de viktigaste idéerna ur det här meddelandet.", 
    "it": "  Estrai le idee principali da questo messaggio.", 
    "id": "  Ekstrak ide utama dari pesan ini."
}

@dataclass
class DefaultMapping:
    encoder : str = 'cuda:1'
    decoder : str = 'cuda:1'

def load_model(mapping = DefaultMapping):
    model = whisper.load_model("large", device='cpu')
    model.eval()
    model.encoder.to(mapping.encoder);
    model.decoder.to(mapping.decoder);

    return model

def empty_cache(f):
    def inner(*args, **kwargs):
        res = f(*args, **kwargs)
        torch.cuda.empty_cache()
        return res
    return inner

def delete_file(arg_idx=0):
    def inner_1(f):
        def inner(*args, **kwargs):
            res = f(*args, **kwargs)
            if os.path.exists(args[arg_idx]):
                os.remove(args[arg_idx])
            return res
        return inner
    return inner_1

@empty_cache
@torch.no_grad()
@delete_file(arg_idx=1)
def speech_to_text(model, audio_fname, mapping=DefaultMapping):
    # audio = whisper.load_audio(audio_fname)
    # audio = whisper.pad_or_trim(audio)
    # spectro = whisper.log_mel_spectrogram(audio)
    # print(spectro.shape)
    # segment = model.encoder(spectro[None].to(mapping.encoder))

    # options = DecodingOptions(
    #             sample_len=1024,
    #         )
    # results = model.decode(segment.to(mapping.decoder), options)

    # language = results[0].language
    # tokenizer = get_tokenizer(language, task="transcribe")
    # result = ''.join([tokenizer.decode(result.tokens).strip() for result in results])
    # print(f"{len(results)} observations : {result}")
    transcription = model.transcribe(audio_fname)
    return transcription['text'], transcription['language']

def text_to_summary(text, language='en'):
    openai.api_key = "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"

    task = tasks.get(language, tasks['en'])

    prompt = task+text

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=512
        )

    return response.choices[0].text.strip()

def speech_to_summary_conditional(model, audio_fname, length_condition, mapping=DefaultMapping):
    text, language = speech_to_text(model, audio_fname, mapping)
    if len(text) < length_condition:
        return text
    return text_to_summary(text, language)
