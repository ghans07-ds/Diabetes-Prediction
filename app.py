import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pickle','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    int_features=[x for x in request.form.values() if x!='']
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=prediction[0]
    
    return render_template('index.html',prediction_text='dibetes: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)