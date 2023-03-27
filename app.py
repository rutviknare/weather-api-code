from flask import Flask, request, jsonify
from urllib.request import urlopen
from apscheduler.schedulers.background import BackgroundScheduler
from collections import Counter
import json
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


def get_weather_conditions_from_api():
    kolhapur_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/kolhapur/today?unitGroup=metric&key=PNSZ5F63HRRRB3P67KPBWMLDM&contentType=json"
    sangli_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/sangli/today?unitGroup=metric&key=PNSZ5F63HRRRB3P67KPBWMLDM&contentType=json"
    nashik_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/nashik/today?unitGroup=metric&key=PNSZ5F63HRRRB3P67KPBWMLDM&contentType=json"

    kolhapur_response = urlopen(kolhapur_url)
    sangli_response = urlopen(sangli_url)
    nashik_response = urlopen(nashik_url)

    kolhapur_data = json.loads(kolhapur_response.read())
    sangli_data = json.loads(sangli_response.read())
    nashik_data = json.loads(nashik_response.read())

    kolhapur_json_obj = json.dumps(kolhapur_data, indent=4)
    sangli_json_obj = json.dumps(sangli_data, indent=4)
    nashik_json_obj = json.dumps(nashik_data, indent=4)

    with open("Kolhapur_weather_data.json", "w") as outfile:
        outfile.write(kolhapur_json_obj)

    with open("Sangli_weather_data.json", "w") as outfile:
        outfile.write(sangli_json_obj)

    with open("Nashik_weather_data.json", "w") as outfile:
        outfile.write(nashik_json_obj)


scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(get_weather_conditions_from_api, 'interval', hours=24)
scheduler.start()

app = Flask(__name__)


@app.route('/currentWeather', methods=['POST'])
def get_current_weather_conditions():
    location = request.form.get('location')
    hr = int(request.form.get('hour'))

    file = None

    if location == 'kolhapur':
        file = open('Kolhapur_weather_data.json')
    elif location == 'sangli':
        file = open('Sangli_weather_data.json')
    elif location == 'nashik':
        file = open('Nashik_weather_data.json')

    data = json.load(file)

    result = dict()
    result['latitude'] = data['latitude']
    result['longitude'] = data['longitude']
    result['address'] = data['resolvedAddress']
    result['description'] = data['description']
    result['temp'] = data['days'][0]['hours'][hr]['temp']
    result['humidity'] = data['days'][0]['hours'][hr]['humidity']
    result['precip'] = data['days'][0]['hours'][hr]['precip']
    result['windspeed'] = data['days'][0]['hours'][hr]['windspeed']
    result['conditions'] = data['days'][0]['hours'][hr]['conditions']

    return jsonify(result)


@app.route('/predictCrop', methods=['POST'])
def predict_crop():
    ph = request.form.get('ph')
    start_date = request.form.get('startDate')
    duration = int(request.form.get('duration'))
    location = request.form.get('location')

    predictions = []

    df = pd.read_csv(location + '.csv')
    start_index = int(df.loc[df['date'] == start_date]['index'])
    for i in range(start_index, (start_index + (duration * 30))):
        index = i % 365
        input_query = np.array([[df.iloc[index]['temp'], df.iloc[index]['humidity'], ph, df.iloc[index]['precip']]])
        input_query = scaler.transform(input_query)
        predictions.append(model.predict(input_query)[0])

    counts = Counter(predictions)
    counts = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    suitable_crops = [count[0] for count in counts]

    return jsonify({'Suitable Crop': suitable_crops[:3]})


if __name__ == '__main__':
    app.run(debug=True)
