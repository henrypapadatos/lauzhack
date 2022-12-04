#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 December 03, 18:58:31
@last modified : 2022 December 04, 13:22:52
"""

import os
from telegram import Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from summarizer import load_model, speech_to_text, speech_to_summary_conditional, DefaultMapping

def start(update, context):
    update.message.reply_text("Hello bad boy! Send /help to get started")

def help(update, context):
    update.message.reply_text("Simply send me a voice message and I will summarize it for you! You can also interact with it on https://ml.romaingrx.com")

def summarize_voice(update, context, model):
    v = update.message.voice
    file = context.bot.get_file(v.file_id)

    file.download("tmp.ogg")

    summary = speech_to_summary_conditional(model, "tmp.ogg", 200)

    update.message.reply_text(summary)


if __name__ == '__main__':
    token = os.environ["TELEGRAM_TOKEN"]
    bob = Bot(token)
    updater = Updater(token, use_context=True)
    
    mapping = DefaultMapping('cuda:1', 'cuda:1')

    model = load_model(mapping)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help))
    dispatcher.add_handler(MessageHandler(Filters.voice, lambda *args, **kwargs : summarize_voice(*args, model=model, **kwargs)))

    updater.start_polling()
