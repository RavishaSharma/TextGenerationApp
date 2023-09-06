
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

tokenizer=pickle.load(open('tokenizer.pkl','rb'))
model = load_model('your_model.h5') 

word_index_reverse = {}
for k,v in tokenizer.word_index.items():
    word_index_reverse[v] = k


@app.route('/')
def hello_world():
    return render_template("PredictiveKeyboard.html")

def preprocess_and_vectorize(example_text):
    
    example_text = re.sub("[^a-zA-Z]"," ", example_text)
    example_text = example_text.lower()
    example_text = tokenizer.texts_to_sequences([example_text]) #text_to_sequences expects a list
    example_text = pad_sequences(example_text,maxlen)[0]
    example_text = example_text.tolist()

    return example_text

def generate_next_n_words(test_sequence, num_words_to_predict):

    for i in range(num_words_to_predict):
        X_test = np.expand_dims(np.array(test_sequence), axis=0)
        preds = model.predict(X_test)[0]
        next_index = np.argmax(preds)
        next_word = word_index_reverse[next_index]
        generated.append(next_word)
        test_sequence.append(next_index)
        test_sequence = test_sequence[1:]
        print(next_word)

    print("Generated Sequence: ", ' '.join(generated))
    print(generated)
    return generated


@app.route('/predict',methods=['POST','GET'])
def predict():

    inputs = [ x for x in request.form.values()]
    text = inputs[0]
    num_of_words = int(inputs[1])
    text = preprocess_and_vectorize(text)
    generated = []
    sequence = generate_next_n_words(text,num_of_words)

    return render_template('PredictiveKeyboard.html',pred = 'Output is {}'.format(sequence))





    '''
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('forest_fire.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest_fire.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")
'''

if __name__ == '__main__':
    app.run(debug=True)
