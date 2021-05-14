import flask
from flask import Flask, render_template, request
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np
app = Flask(__name__)
model = pickle.load(open("LR2.pkl", "rb"))


@app.route("/")
def main():
    return render_template("form.html")


@app.route('/result', methods=['POST'])
def result():
    data1 = request.form['Age']
    data2 = request.form['Gender']
    data3 = request.form['family_history']
    data4 = request.form['benefits']
    data5 = request.form['care_options']
    data6 = request.form['anonymity']
    data7 = request.form['leave']
    data8 = request.form['work_interfere']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
    pred = model.predict(arr)
    return render_template('result.html',result=pred)

if __name__ == "__main__":
    app.run(debug=True)

