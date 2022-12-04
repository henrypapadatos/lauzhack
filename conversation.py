import openai

class Conversation:

    answer_task = {
    "en": "Given those previous explanations, ", 
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

    subject_task = {
    "en": "Given those previous explanations, enumerate the important topics and order them starting with the most important one", 
    "zh": "  从这条消息中提取主要思想。", 
    "de": "  Holen Sie de wichtigsten Ideen aus dieser Nachricht heraus.", 
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

    question_task = {
    "en": lambda x: "Given those previous explanations, ask me questions about {} to evaluate my understanding of the explanations".format(x), 
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

    def __init__(self,explanations:str,
                 language:str="en") -> None:
        openai.api_key = "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"
        self.explanations:str = explanations
        self.language:str = language
        self.conversation:list = []
        self.subject_list:list = []

    def call_gpt(self,prompt:str) -> str:
        output = openai.Completion.create(engine="text-davinci-003",
                                          prompt=prompt,
                                          temperature=0.0,
                                          max_tokens=512
                                          ).choices[0].text.replace("\n","",2)
        return output

    def ask_question(self,question:str) -> str:
        task = self.answer_task.get(self.language, self.answer_task['en'])
        prompt = self.explanations + " \\n \\n" + task + question
        answer = self.call_gpt(prompt)
        self.conversation.append(question)
        self.conversation.append(answer)
        return answer

    def generate_subject_list(self) -> None:
        task = self.subject_task.get(self.language, self.subject_task['en'])
        prompt = self.explanations + "\\n" + task
        subjects = self.call_gpt(prompt)
        subject_list_raw = subjects.split("\n")
        self.subject_list = [subject.split(" ",1)[1] for subject in subject_list_raw]

    def evaluate_understanding(self) -> str:
        if len(self.subject_list) == 0:
            self.generate_subject_list()
        subject = self.subject_list.pop(0)
        task = self.question_task.get(self.language, self.question_task['en'])(subject)
        prompt = self.explanations + "\\n" + task
        question = self.call_gpt(prompt)
        self.conversation.append(question)
        return question