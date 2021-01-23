from wsgiref import simple_server
from flask import Flask, request, render_template
import pickle
import json
import numpy as np
"""
*****************************************************************************
*
* filename:       main.py
* version:        1.0
* author:         HARISH
* creation date:  23-jan-2021
*
* change history:
*
* who             when           version  change (include bug# if apply)
* ----------      -----------    -------  ------------------------------
* HARISH          23-JAN-2021    1.0      initial creation
*
*
* description:    flask main file to run application
*
****************************************************************************
"""

app = Flask(__name__)

def get_predict_weather(landaveragetemperature, landmaxtemperature, landmintemperature):
    """
    * method: get_predict_profit
    * description: method to predict the results
    * return: prediction result
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * HARISH          23-JAN-2021    1.0      initial creation
    *
    * Parameters
    *
    """
    with open('models/weather_prediction.pkl', 'rb') as f:
        model = pickle.load(f)

    with open("models/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']

    x = np.zeros(len(data_columns))
    x = np.reshape(x, newshape=(1, len(data_columns)), order='C')
    x[0][0] = landaveragetemperature
    x[0][1] = landmaxtemperature
    x[0][2] = landmintemperature


    return model.predict(x)

@app.route('/')
def index_page():
    """
    * method: index_page
    * description: method to call index html page
    * return: index.html
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * HARISH          23-JAN-2021    1.0      initial creation
    *
    * Parameters
    *   None
    """
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    """
    * method: predict
    * description: method to predict
    * return: index.html
    *
    * who             when           version  change (include bug# if apply)
    * ----------      -----------    -------  ------------------------------
    * HARISH          20-JAN-2021    1.0      initial creation
    *
    * Parameters
    *   None
    """
    if request.method == 'POST':
        landaveragetemperature = request.form['landaveragetemperature']
        landmaxtemperature = request.form["landmaxtemperature"]
        landmintemperature = request.form["landmintemperature"]

        output = get_predict_weather(landaveragetemperature, landmaxtemperature, landmintemperature)
        return render_template('index.html',show_hidden=True, prediction_text='This Project is done by Harish Musti and temparature must be  {}'.format(output))


if __name__ == "__main__":
    #app.run(debug=True)
    host = '0.0.0.0'
    port = 5008
    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()
