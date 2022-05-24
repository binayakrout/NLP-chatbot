from flask import Flask, render_template, request
from dialogue import get_dialogue

app = Flask(__name__)

chat_history = []

@app.route("/", methods=['GET'])
def index():
    if request.method == 'GET':
        return render_template("index.html")

@app.route("/chat", methods=['GET','POST'])
def chat():
    if request.method == 'POST':
        json_data = request.get_json()
        data = json_data['msg']
        res = get_dialogue()
        chat_history.append(data)
        return res
    

@app.route("/quit", methods=['POST'])
def quit():
    if request.method == 'POST':
        print(chat_history)
        chat_history.clear()
        return render_template("final.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)