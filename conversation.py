import openai
from functools import lru_cache
from typing import Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Conversation:

    answer_task = {
    "en": "  Given those previous explanations, ", 
    "zh": "  鉴于之前的那些解释。", 
    "de": "  In Anbetracht dieser vorherigen Erklärungen,", 
    "es": "  Dadas las explicaciones anteriores,", 
    "ru": "  Учитывая предыдущие объяснения,",
    "fr": "  Etant donné les explications précédentes", 
    "ja": "  そういったこれまでの説明を踏まえると, ", 
    "pt": "  Dadas essas explicações anteriores, ", 
    "tr": "  Önceki açıklamalar göz önüne alındığında, ", 
    "pl": "  Biorąc pod uwagę te wcześniejsze wyjaśnienia, ", 
    "nl": "  Gezien deze eerdere verklaringen, ",
    "sv": "  Vzhľadom na tieto predchádzajúce vysvetlenia, ", 
    "it": "  Date le spiegazioni precedenti, ", 
    "id": "  Mengingat penjelasan-penjelasan sebelumnya, "
    }

    subject_task = {
    "en": "  Given those previous explanations, enumerate the important topics and order them starting with the most important one", 
    "zh": "  鉴于前面的解释，请列举出重要的议题，并从最重要的议题开始排序, ", 
    "de": "  Zählen Sie anhand der vorangegangenen Erläuterungen die wichtigsten Themen auf und ordnen Sie sie, beginnend mit dem wichtigsten Thema, ", 
    "es": "  Teniendo en cuenta las explicaciones anteriores, enumera los temas importantes y ordénalos empezando por el más importante", 
    "ru": "  Учитывая предыдущие объяснения, перечислите важные темы и упорядочьте их, начиная с самой важной", 
    "fr": "  Compte tenu de ces explications, énumérez les sujets importants et classez-les en commençant par le plus important", 
    "ja": "  これまでの説明を踏まえて、重要なトピックを列挙し、最も重要なものから順番に説明しなさい ", 
    "pt": "  Dadas essas explicações prévias, enumerar os tópicos importantes e ordená-los começando pelo mais importante ", 
    "tr": "  Bu mesajdan ana fikirleri çikarin, ", 
    "pl": "  Önceki açıklamaları göz önünde bulundurarak, önemli konuları sıralayın ve en önemlisinden başlayarak sıralayın, ", 
    "nl": "  Op basis van deze voorgaande uiteenzettingen, de belangrijke onderwerpen opnoemen en ze ordenen, te beginnen met het belangrijkste onderwerp, ",
    "sv": "  Glede na prejšnja pojasnila naštejte pomembne teme in jih razvrstite po vrstnem redu, začenši z najpomembnejšo, ", 
    "it": "  Date le spiegazioni precedenti, elencate gli argomenti importanti e ordinateli a partire dal più importante, ", 
    "id": "  Berdasarkan penjelasan sebelumnya, sebutkan topik-topik penting dan urutkan mulai dari yang paling penting, "
    }

    question_task = {
    "en": lambda x: "Given those previous explanations, ask me one question about {} to evaluate my understanding of the explanations".format(x),
    "zh": lambda x: "  鉴于之前的解释，向我提出关于以下方面的问题 {} 以评估我对解释的理解。".format(x),
    "de": lambda x: "  Stellen Sie mir angesichts der vorangegangenen Erklärungen Fragen zu {}, um mein Verständnis der Erklärungen zu bewerten".format(x),
    "es": lambda x: "  Dadas esas explicaciones previas, hágame preguntas sobre {} para evaluar mi comprensión de las explicaciones.".format(x),
    "ru": lambda x: "  Учитывая предыдущие объяснения, задайте мне вопросы о {}, чтобы оценить мое понимание объяснений".format(x),
    "ko": lambda x: "  이 메시지에서 주요 아이디어를 추출하십시오.".format(x),
    "fr": lambda x: "  Compte tenu de ces explications précédentes, posez-moi des questions sur {} pour évaluer ma compréhension".format(x),
    "ja": lambda x: "  それらの前の説明を踏まえて、{}について質問して、私の説明の理解度を評価してください。".format(x),
    "pt": lambda x: "  Dadas as explicações anteriores, fazer-me perguntas sobre {} para avaliar a minha compreensão das explicações.".format(x),
    "tr": lambda x: "  Önceki açıklamaları göz önünde bulundurarak, açıklamaları anladığımı değerlendirmek için bana {} hakkında sorular sorun.".format(x),
    "pl": lambda x: "  Biorąc pod uwagę poprzednie wyjaśnienia, zadaj mi pytania dotyczące {}, aby ocenić moje zrozumienie wyjaśnień".format(x),
    "nl": lambda x: "  Stel me, gezien deze eerdere uitleg, vragen over {} om te evalueren of ik de uitleg begrijp..".format(x),
    "sv": lambda x: "  Med tanke på dessa tidigare förklaringar, ställ frågor om {} för att utvärdera min förståelse av förklaringarna..".format(x),
    "it": lambda x: "  Date le spiegazioni precedenti, fatemi domande su {} per valutare la mia comprensione delle spiegazioni.".format(x),
    "id": lambda x: "  Mengingat penjelasan-penjelasan sebelumnya, ajukan pertanyaan kepada saya tentang {} untuk mengevaluasi pemahaman saya".format(x)
    }

    evaluation_task = {
    "en": lambda x, y: "Given those previous explanations, I answered {} to this question: {}. Tell me first if I am right or wrong and then give the right answer of the question".format(x,y),
    "zh": lambda x, y:"  鉴于之前的这些解释，我对这个问题的回答是{}。{}. 首先告诉我，我是对还是错，然后给出问题的正确答案.".format(x,y),
    "de": lambda x, y:"  In Anbetracht der vorangegangenen Erklärungen habe ich {} auf diese Frage geantwortet: {}. Sagen Sie mir zuerst, ob ich richtig oder falsch liege und geben Sie dann die richtige Antwort auf die Frage.".format(x,y),
    "es": lambda x, y:"  Dadas esas explicaciones anteriores, respondí {} a esta pregunta: {}. Dígame primero si tengo razón o no y luego dé la respuesta correcta de la pregunta.".format(x,y),
    "ru": lambda x, y:"  Учитывая эти предыдущие объяснения, я ответил {} на этот вопрос: {}. Скажите сначала, прав я или нет, а затем дайте правильный ответ на вопрос.".format(x,y),
    "fr": lambda x, y:"  Compte tenu de ces explications précédentes, j'ai répondu {} à cette question : {}. Dites-moi d'abord si j'ai raison ou tort et ensuite donnez la bonne réponse à la question.".format(x,y),
    "ja": lambda x, y:"  それらのこれまでの説明から、この質問には｛｝と答えました。{}. まず私が正しいのか間違っているのかを教えてください、そして質問の正しい答えを教えてください ".format(x,y),
    "pt": lambda x, y:"  Dadas estas explicações anteriores, respondi {} a esta pergunta: {}. Digam-me primeiro se estou certo ou errado e depois dêem a resposta certa à pergunta.".format(x,y),
    "tr": lambda x, y:"  Önceki açıklamaları göz önünde bulundurarak bu soruya {} cevabını verdim: {}. Önce bana doğru mu yanlış mı olduğumu söyleyin ve sonra sorunun doğru cevabını verin.".format(x,y),
    "pl": lambda x, y:"  Biorąc pod uwagę te wcześniejsze wyjaśnienia, na to pytanie odpowiedziałem {}: {}. Powiedz mi najpierw, czy mam rację, czy nie, a następnie podaj właściwą odpowiedź na pytanie.".format(x,y),
    "nl": lambda x, y:"  Gezien deze eerdere uitleg, antwoordde ik {} op deze vraag: {}. Zeg eerst of ik gelijk heb of niet en geef dan het juiste antwoord op de vraag.".format(x,y),
    "sv": lambda x, y:"  Med tanke på dessa tidigare förklaringar svarade jag {} på denna fråga: {}. Säg först om jag har rätt eller fel och ge sedan det rätta svaret på frågan".format(x,y),
    "it": lambda x, y:"  Date le spiegazioni precedenti, ho risposto {} a questa domanda: {}. Ditemi prima se ho ragione o torto e poi date la giusta risposta alla domanda ".format(x,y),
    "id": lambda x, y:"  Dengan penjelasan-penjelasan sebelumnya, saya menjawab {} untuk pertanyaan ini: {}. Beritahu saya terlebih dahulu apakah saya benar atau salah dan kemudian berikan jawaban yang benar dari pertanyaan tersebut".format(x,y),
    }

    def __init__(self,explanations:str,
                 language:str="en") -> None:
        openai.api_key = "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"
        self.explanations:str = explanations
        self.language:str = language
        self.conversation:list = []
        self.subject_list:list = []
        self.model: str = model

    @staticmethod
    @lru_cache(maxsize=1)
    def load_bloomz_model(device: str = "cuda:1") -> Union[AutoTokenizer, AutoModelForCausalLM]:
        checkpoint = 'bigscience/bloomz-3b'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        device_map = 'balanced_low_0' if device not in ['cpu', 'cuda:0'] else "auto"
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map=device_map).to(device)
        return tokenizer, model

    def call_gpt(self, prompt:str) -> str:
        output = openai.Completion.create(engine="text-davinci-003",
                                          prompt=prompt,
                                          temperature=0.0,
                                          max_tokens=512
                                          ).choices[0].text.replace("\n","",2)
        return output

    def call_bloomz(self, prompt:str) -> str:
        device = 'cuda:1'
        bloomz_tokenizer, bloomz_model = Conversation.load_bloomz_model(device)
        inputs = bloomz_tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_tokens = bloomz_model.generate(inputs, max_length=500)
        output = bloomz_tokenizer.decode(output_tokens[0])
        return output

    def call_model(self, prompt:str) -> str:
        if self.model == 'gpt3':
            return self.call_gpt(prompt)
        else:
            return self.call_bloomz(prompt)

    def ask_question(self,question:str) -> str:
        task = self.answer_task.get(self.language, self.answer_task['en'])
        prompt = self.explanations + " \\n \\n" + task + question
        answer = self.call_model(prompt)
        self.conversation.append(question)
        self.conversation.append(answer)
        return answer

    def generate_subject_list(self) -> None:
        task = self.subject_task.get(self.language, self.subject_task['en'])
        prompt = self.explanations + "\\n" + task
        subjects = self.call_model(prompt)
        subject_list_raw = subjects.split("\n")
        self.subject_list = [subject.split(" ",1)[1] for subject in subject_list_raw]

    def evaluate_understanding(self) -> str:
        if len(self.subject_list) == 0:
            self.generate_subject_list()
        subject = self.subject_list.pop(0)
        task = self.question_task.get(self.language, self.question_task['en'])(subject)
        prompt = self.explanations + " \\n \\n" + task
        question = self.call_model(prompt)
        self.conversation.append(question)
        return question

    def evalutate_answer(self,question:str,answer:str) -> str:
        task = self.evaluation_task.get(self.language, self.evaluation_task['en'])(answer,question)
        prompt = self.explanations + " \\n \\n" + task
        correction = self.call_model(prompt)
        if question != self.conversation[-1]:
            self.conversation.append(question)
        self.conversation.append(answer)
        self.conversation.append(correction)
        return correction
