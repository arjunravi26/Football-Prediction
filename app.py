from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
import pickle
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

app = Flask(__name__)

# Your existing route for rendering the form


@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')


# Your existing route for handling form submission
@app.route('/submit', methods=['POST'])
def submit_form():
    # Access form data using request.form
    hometeam = request.form.get('hometeam')
    awayteam = request.form.get('awayteam')
    htp = request.form.get('htp')
    atp = request.form.get('atp')
    hm1 = request.form.get('hm1')
    hm2 = request.form.get('hm2')
    hm3 = request.form.get('hm3')
    am1 = request.form.get('am1')
    am2 = request.form.get('am2')
    am3 = request.form.get('am3')
    htlp = request.form.get('htlp')
    atlp = request.form.get('atlp')
    # Preprocess the form data
    user_input_df = pd.DataFrame({
        'HomeTeam': [hometeam],
        'AwayTeam': [awayteam],
        'HTP': [int(htp)],
        'ATP': [int(atp)],
        'HM1': [hm1[0]],
        'HM2': [hm2[0]],
        'HM3': [hm3[0]],
        'AM1': [am1[0]],
        'AM2': [am2[0]],
        'AM3': [am3[0]],
        'DiffFormPts': [int(htp)-int(atp)],
        'DiffLP': [int(htlp)-int(atlp)]
    })
    data = user_input_df.copy()
    data.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
  
    
    # data_scale = scaler.fit_transform(data_scale)
    # data_scale_scaled = scaler.fit_transform([data_scale])[0]
    data.loc[0, ['HTP', 'ATP']] = data.iloc[:2, data.columns.get_indexer(['HTP', 'ATP'])].values.flatten()
    data['DiffFormPts'] = data['HTP'] - data['ATP']
    
    def encode_data(input_data):
        ordinal_encoder = OrdinalEncoder(categories=[['L', 'D', 'W']], dtype=int)
        cols_encode = ['HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3']
        for col in cols_encode:
            input_data[col] = ordinal_encoder.fit_transform(
                input_data[[col]])
        return input_data
    data = encode_data(data)
    data_type_mapping = {'HTP': float, 'ATP': float, 'DiffFormPts': float,
                     'DiffLP': 'int64', 'HM1': 'int64', 'HM2': 'int64', 'HM3': 'int64',
                     'AM1': 'int64', 'AM2': 'int64', 'AM3': 'int64'}

    data = data.astype(data_type_mapping)
    scaler= pickle.load(open('scaler1.pkl', "rb"))
    data_scale = scaler.transform(data)
    rfc = pickle.load(open('rfc.pkl', "rb"))
    user_pred = rfc.predict(data_scale)

    # return f"Form submitted successfully! Predicted Output: {user_pred}"
    team = user_input_df.at[0, 'HomeTeam']
    if user_pred == 0:
        text = f"“Based on current predictions, {team} is more likely to winning.”"

    else:
        text = f"“Based on current predictions, {team} is more likely to either draw or suffer a defeat.”"

    return render_template('result.html', result_text=text)


if __name__ == '__main__':
    app.run(debug=True)
