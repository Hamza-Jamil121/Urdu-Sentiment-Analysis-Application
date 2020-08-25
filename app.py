from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
import pickle

# load the model from disk

clf = pickle.load(open('urdu_Modell.pkl', 'rb'))
cv=pickle.load(open('urdu_counter1.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('urdu.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect=cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        if my_prediction ==0:
            res_val = "Negative"
        
        elif  my_prediction==1:
            
            res_val='Neutral'
        else:
            res_val='Positve'
       
    
    return render_template('urdu.html',prediction_text=' Sentence is {}'.format(res_val))
if __name__ == '__main__':
	app.run(debug=True)