from flask import Flask,request,render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

app = Flask( __name__)

@app.route('/',methods=['GET'])
def  home(): 
    return render_template("home1.html")

model = load_model('chatbot.h5')
tokenizer = pickle.load(open('tokenizer.pkl','rb'))
le = pickle.load(open('labelencoder.pkl','rb'))

with open('intents.json') as file:
  data = json.load(file)

@app.route('/',methods=['POST'])
def predict():
    text = request.form.get('user')
    text = tokenizer.texts_to_sequences([text])
    text = pad_sequences(text,maxlen=20)
    result = model.predict(text)
    output_class = np.argmax(result)
    output = le.inverse_transform(np.array([output_class]))
    for example in data['intents']:
      if example['tag'] == output[0]:
        prediction_text = np.random.choice(example['responses'])
    return render_template('home1.html', prediction_text=prediction_text)

if __name__=='__main__':
    app.run()