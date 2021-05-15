from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    features1=[int(x) for x in request.form.values()]
    features1.pop()
    final=[np.array(features1)]
    print(features1)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)
    if(float(output)>0.5):
        return render_template('index.html',prediction='félicitation il y a de fortes chances que votre crédit soit approuvé')
    else:
        return render_template('index.html',prediction="dommage votre crédit est suscibtible de ne pas être approuvé")


if __name__ == '__main__':
    app.run(debug=True)
