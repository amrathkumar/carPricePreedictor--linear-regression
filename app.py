from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)

mined = pd.read_csv('mined.csv')
model = pickle.load(open('LR_model.pkl','rb'))
print(mined.columns)
@app.route('/')
def index():
    companies = sorted(mined["company"].unique().tolist())
    car_model = sorted(mined["name"].unique().tolist())
    year = sorted(mined["year"].unique().tolist())
    fuel_type = sorted(mined["fuel_type"].unique().tolist())

    car_dict = {}

    for company in companies:
        car_dict[company] = list(mined[mined['company'] == company]['name'])
    return render_template('index.html',
        company  = companies,
        year =year,
        fuel_type = fuel_type,
        car_dict = car_dict
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        kms = request.form.get('kilo_driven')

        # 🔥 DEBUG PRINT (IMPORTANT)
        print(company, car_model, year, fuel_type, kms)

        # ✅ Convert safely
        year = int(year)
        kms = int(kms)

        input_df = pd.DataFrame(
            [[car_model, company, year, fuel_type, kms, 0]],
            columns=['name', 'company', 'year', 'fuel_type', 'km', 'index']
        )

        print(input_df)  # 👈 CHECK THIS OUTPUT

        prediction = model.predict(input_df)

        return str(round(prediction[0], 2))

    except Exception as e:
        print("ERROR:", e)
        return str(e)


if __name__ == '__main__':
    app.run()
