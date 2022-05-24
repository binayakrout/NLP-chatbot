from flask import Flask, render_template, request
from dialogue import get_dialogue
from get_user_intent import get_intent
from customNER import get_ner
from check_history import check_update

import logging

app = Flask(__name__)

logging.basicConfig(filename='atis_details_bot.log', level=logging.DEBUG)
logger = logging.getLogger("main")

chat_history = []

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

@app.route("/", methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

@app.route("/chat", methods=['POST'])
def chat():
    if request.method == 'POST':
        logger.info("Received input data")
        json_data = request.get_json()
        input_data = json_data['msg']
        logger.info("Fetching Intent of the input data: "+input_data)
        intent_data = get_intent(input_data)
        print(intent_data)
        logger.info("Fetching Named-Entities of the input data: "+input_data)
        ner_data = get_ner(input_data)
        print(ner_data)
        data = Merge(intent_data, ner_data)
        new_data = check_update(data,chat_history)
        print(new_data)
        logger.info("Fetching final response of the input data: "+input_data)
        res = get_dialogue(new_data)
        dialogue_res = {'response': res}
        chat_history.append(Merge(data, dialogue_res))
        logger.info("Final response of the input data: "+res)
        return res
    

@app.route("/quit", methods=['POST'])
def quit():
    if request.method == 'POST':
        chat_history.clear()
        return render_template("final.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)