from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from model import load_model, predict

app = Flask(__name__)
# load_model now returns (kind, model, scaler)
model = load_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # collect 30 features from form; assume names feature0..feature29
        try:
            features = [float(request.form.get(f'feature{i}', 0)) for i in range(30)]
        except ValueError:
            return render_template('index.html', error='Please enter valid numbers for all features.')
        proba, pred = predict(model, np.array(features).reshape(1, -1))
        return render_template('result.html', proba=round(float(proba), 4), pred=int(pred))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
