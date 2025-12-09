import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__, template_folder='templates')

MODEL_PATH = 'bangalore_house_price_model.pkl'
LOCATIONS_TO_KEEP_PATH = 'locations_to_keep.pkl'

best_model = None
locations_to_keep = None

try:
    best_model = joblib.load(MODEL_PATH)
    locations_to_keep = joblib.load(LOCATIONS_TO_KEEP_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Locations to keep loaded from {LOCATIONS_TO_KEEP_PATH}")

except FileNotFoundError:
    print(f"Error: Model file ({MODEL_PATH}) or locations file ({LOCATIONS_TO_KEEP_PATH}) not found.")
    print("Please run the training script first to generate these files.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")

amenity_columns_for_9_to_1_transform = [
    'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack',
    'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility',
    'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup', 'CarParking',
    'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
    'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 'LiftAvailable', 'BED',
    'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa',
    'Wardrobe', 'Stadium'
]

expected_X_columns_for_df = [
    'Area', 'No. of Bedrooms',
    'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack',
    'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom', 'SportsFacility',
    'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup', 'CarParking',
    'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 'WashingMachine',
    'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 'LiftAvailable', 'BED',
    'VaastuCompliant', 'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa',
    'Wardrobe', 'Stadium', 'Resale',
    'Location_grouped'
]

@app.route('/')
def serve_intro():
    return send_file('index0.html')
@app.route('/predict_form')
def serve_index():
    return send_file('index.html')


@app.route('/get_locations', methods=['GET'])
def get_locations():
    if locations_to_keep is None:
        return jsonify({"error": "Locations data not loaded."}), 500
    return jsonify({"locations": sorted(list(locations_to_keep))})

@app.route('/predict', methods=['POST'])
def predict():
    global best_model, locations_to_keep

    if best_model is None or locations_to_keep is None:
        return jsonify({"error": "Model or locations data not loaded. Server might be misconfigured. Check server logs."}), 500

    try:
        data = request.get_json(force=True)

        single_prediction_row = {}
        for col in expected_X_columns_for_df:
            if col == 'Area':
                single_prediction_row[col] = float(data.get(col))
            elif col == 'No. of Bedrooms':
                single_prediction_row[col] = int(data.get(col))
            elif col == 'Location_grouped':
                single_prediction_row[col] = None
            else:
                single_prediction_row[col] = int(data.get(col, 0))

        raw_location = data.get('Location')
        if raw_location in locations_to_keep:
            single_prediction_row['Location_grouped'] = raw_location
        else:
            single_prediction_row['Location_grouped'] = 'Other'

        for col in amenity_columns_for_9_to_1_transform:
            if col in single_prediction_row and single_prediction_row[col] == 9:
                single_prediction_row[col] = 1

        prediction_df = pd.DataFrame([single_prediction_row], columns=expected_X_columns_for_df)

        print("\nPrediction DataFrame prepared:")
        print(prediction_df)

        y_pred_log = best_model.predict(prediction_df)
        y_pred_original = np.expm1(y_pred_log)

        return jsonify({'predicted_price': round(float(y_pred_original[0]), 2)})

    except KeyError as ke:
        return jsonify({"error": f"Missing data for required field: {ke}. Please ensure 'Area', 'No. of Bedrooms', and 'Location' are provided, and all other amenity fields are correctly named if sent."}), 400
    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {ve}. Please check the format of numbers (Area, Bedrooms)."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':

    app.run(debug=True, host='127.0.0.1', port=5000)

