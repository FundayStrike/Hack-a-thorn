from flask import Flask, request, render_template
import pickle
# import tensorflow.keras as keras

# model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form.get('text')
        # include below code after properly integrating model to flask
        # pred = model.predict([text])
        # print(pred)
        return render_template('index.html', text=text)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)