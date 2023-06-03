from flask import Flask, request, render_template, session
from transformers import BertTokenizer, TFBertForSequenceClassification
from summarizer import Summarizer, TransformerSummarizer
import tensorflow as tf
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# initialising the model and tokenizer
tokenizer = BertTokenizer.from_pretrained("8CSI/Travel_Sentimental_Analysis")
model = TFBertForSequenceClassification.from_pretrained("8CSI/Travel_Sentimental_Analysis")

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'prev_text' not in session and 'prev_results' not in session:
        session['prev_text'] = []
        session['prev_results'] = []
    if request.method == 'POST':
        text = request.form.get('text')
        if text == '':
            return render_template('index.html', get_pred=False, prev_text=session['prev_text'], prev_results=session['prev_results'], len_loop=len(session['prev_text']))
        tf_batch = tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors='tf', return_token_type_ids=True, return_attention_mask=True)
        tf_outputs = model(tf_batch.input_ids)
        tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
        # extract positive and negative probabilities
        pos, neg = round(tf_predictions[0][1].numpy()*100, 1), round(tf_predictions[0][0].numpy()*100, 1)
        summary = summarise_review(text)
        if summary == text or summary == '':
            summary = 'Sorry, the inputted review is too short to generate a summary.'
        # save data into session storage
        print('Before', session['prev_text'], session['prev_results'], len(session['prev_text']))
        if len(text) < 300:
            session['prev_text'].append(text)
        else:
            session['prev_text'].append(text[:300]+'...')
        if pos > neg:
            session['prev_results'].append(1) # 1 indicates positive, vice versa.
        else:
            session['prev_results'].append(0)
        session.modified = True
        print('After', session['prev_text'], session['prev_results'], len(session['prev_text']))
        return render_template('index.html', get_pred=True, text=text, pos=pos, neg=neg, summary=summary, prev_text=session['prev_text'], prev_results=session['prev_results'], len_loop=len(session['prev_text']))
    return render_template('index.html', get_pred=False, prev_text=session['prev_text'], prev_results=session['prev_results'], len_loop=len(session['prev_text']))

def summarise_review(text):
    GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    full = ''.join(GPT2_model(text, ratio=0.2)) 
    return full

if __name__ == "__main__":
    app.run(debug=False)
