from flask import Flask,render_template,request
import joblib
import numpy as np
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("form.html")
@app.route("/submit",methods=["post"])
def submit():
    a1=request.form["preg"]
    a2=request.form["gluco"]
    a3=request.form["blood"]
    a4=request.form["skin"]
    a5=request.form["insulin"]
    a6=request.form["BMI"]
    a7=request.form["pdf"]
    a8=request.form["age"]
    model=joblib.load("/home/aman/diabaetics/mode.joblib")
    predict=model.predict(np.array([a1,a2,a3,a4,a5,a6,a7,a8]).reshape(1,-1))
    return str(predict)
app.run()

