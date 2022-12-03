#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2022 December 03, 18:58:31
@last modified : 2022 December 04, 00:45:42
"""

from telegram import Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from summarizer import load_model, speech_to_text, speech_to_summary_conditional

def start(update, context):
    update.message.reply_text("Hello bad boy!")

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

    model = load_model()

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.voice, lambda *args, **kwargs : summarize_voice(*args, model=model, **kwargs)))

    updater.start_polling()
