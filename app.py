from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators
import numpy as np
import joblib

# 学習済みモデルを読み込む
def predict(parameters):
    
    pos = joblib.load('bats.pkl')
    pred = 0
    
    for i in range(len(pos)):
        pred += parameters[i] * pos[i]
        
    # 定数項
    pred += 30

    return pred

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'


class IrisForm(Form):
    X1 = FloatField("町ごとの一人当たりの犯罪率 [0, 1]",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=1)])

    X2  = FloatField("25000平方フィートを超える敷地に区画されている比 [0, 20]",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=20)])

    X3 = FloatField("町当たりの非小売業メーカーの割合 [0, 30]",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=30)])

    X4  = FloatField("1 住戸当たりの平均部屋数 [0, 10]",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])

    X5  = FloatField("10000 ドル当たりの固定資産税の合計 [0, 500]",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=500)])

    submit = SubmitField("判定")

@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = IrisForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('index.html', form=form)
        else:            
            X1 = float(request.form["X1"])            
            X2  = float(request.form["X2"])            
            X3 = float(request.form["X3"])            
            X4  = float(request.form["X4"])
            X5  = float(request.form["X5"])

            x = np.array([X1, X2, X3, X4, X5])
            pred = predict(x)

            return render_template('result.html', Predict=pred)
    elif request.method == 'GET':

        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run(debug=True)