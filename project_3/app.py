#This is a deployment for suicidal web apk using Flask
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
app = Flask(__name__)
# On starting to encrypt new values for model training it was realised that 
# the encryption for new single values would be hard for some labels 
# encoded in some hard algorithmic way like LabelEncoder & BinaryEncoder
# So the best way was to assign the data frame categories needed to deal with
# in a new csv file and deal with it hear to use sklearn libraries encryption
# in a direct way. 
df=pd.read_csv('suicide stats.csv')
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
df['generation']=le1.fit_transform(list(df['generation']))
from category_encoders import BinaryEncoder
encoder1= BinaryEncoder(cols=['country'])
# encoder2= BinaryEncoder(cols=['Region'])
encoder1.fit(df['country'])
# encoder2.fit(df['Region'])
country_binary=encoder1.transform(df['country'])
# region_binary=encoder2.transform(df['Region'])
model=joblib.load('model.pkl')
@app.route("/")
def home():
    return render_template ('index.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    sex_male= 1 if request.form['sex'] == 'Male' else 0
    dict_age={'5-14 years':1,'15-24 years':2,'25-34 years':3,'35-54 years':4,'55-74 years':5,'75+ years':6}
    age= dict_age['%s'%(request.form['age-range'])]
    generation=le1.transform(['%s' %(request.form['generation'])])[0]
    index=df[df['country']=='%s' %(request.form['country'])].index[0]
    country_1=country_binary.iloc[index][1]
    country_2=country_binary.iloc[index][2]
    country_3=country_binary.iloc[index][3]
    country_4=country_binary.iloc[index][4]
    country_5=country_binary.iloc[index][5]
    country_6=country_binary.iloc[index][6]
    country_7=country_binary.iloc[index][7]
    index1=df[df['country']=='%s' %(request.form['country'])].index[1]
    prediction=int(model.predict([[country_1, country_2, country_3, country_4, country_5,
       country_6, country_7, age, generation, sex_male]]))
    print(prediction)
    return render_template('index.html',result=prediction)

if __name__ == "__main__":
    app.run()

