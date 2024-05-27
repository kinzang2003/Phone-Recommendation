from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from difflib import get_close_matches

app = Flask(__name__)

# Load the trained model pipeline and label encoder
dt_model_path = 'model/decision_tree_pipeline.pkl'
dt_model = joblib.load(dt_model_path)
dt_label_encoder_path = 'model/label_encoder.pkl'
dt_label_encoder = joblib.load(dt_label_encoder_path)

# Load the trained k-NN model pipeline and label encoder
knn_model_path = 'model/knn_pipeline.pkl'
knn_model = joblib.load(knn_model_path)
knn_label_encoder_path = 'model/label_encoderforknn.pkl'
knn_label_encoder = joblib.load(knn_label_encoder_path)

# Load the Excel file and get unique brands
file_path = 'output.xlsx'  # Update the file path
data = pd.read_excel(file_path)
unique_brands = data['Brand'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html', brands=unique_brands)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data_input = request.form.to_dict()

        # Normalize the brand input to handle case insensitivity
        brand_input = data_input.get('Brand', '').strip().upper()
        closest_match = get_close_matches(brand_input, unique_brands, n=1, cutoff=0.6)
        if closest_match:
            data_input['Brand'] = closest_match[0]
        else:
            return jsonify({'error': 'Brand not recognized. Please try again.'}), 400

        # Convert data to DataFrame
        input_df = pd.DataFrame([data_input])

        # Encode the 'Brand' column
        input_df['Brand'] = dt_label_encoder.transform(input_df['Brand'])

        # Make prediction
        prediction = dt_model.predict(input_df)[0]

        # Get the details of the predicted phone
        phone_details = data[data['Name'] == prediction].to_dict(orient='records')[0]

        return jsonify({'prediction': prediction, 'phone_details': phone_details})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knn_predict', methods=['POST'])
def knn_predict():
    try:
        # Get the data from the POST request
        data_input = request.form.to_dict()

        # Normalize the brand input to handle case insensitivity
        brand_input = data_input.get('Brand', '').strip().upper()
        closest_match = get_close_matches(brand_input, unique_brands, n=1, cutoff=0.6)
        if closest_match:
            data_input['Brand'] = closest_match[0]
        else:
            return jsonify({'error': 'Brand not recognized. Please try again.'}), 400

        # Convert data to DataFrame
        input_df = pd.DataFrame([data_input])

        # Encode the 'Brand' column
        input_df['Brand'] = knn_label_encoder.transform(input_df['Brand'])

        # Scale the data using the pipeline's scaler
        scaled_input_data = knn_model.named_steps['scaler'].transform(input_df)

        # Make prediction using the k-NN model within the pipeline
        knn = knn_model.named_steps['knn']
        distances, indices = knn.kneighbors(scaled_input_data, n_neighbors=5)

        recommended_phones = []
        for idx in indices[0]:
            recommended_phones.append(data.iloc[idx]['Name'])

        return jsonify({'predictions': recommended_phones})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['POST'])
def stats():
    try:
        user_data = request.json

        # Process user data
        user_df = pd.DataFrame([user_data])
        brand_input = user_data.get('Brand', '').strip().upper()
        closest_match = get_close_matches(brand_input, unique_brands, n=1, cutoff=0.6)
        if closest_match:
            user_df['Brand'] = closest_match[0]
        else:
            return jsonify({'error': 'Brand not recognized. Please try again.'}), 400
        
        # Encode the 'Brand' column
        user_df['Brand'] = dt_label_encoder.transform(user_df['Brand'])

        # Predict using the model
        model_prediction = dt_model.predict(user_df)[0]

        # Fetch model output stats from the Excel file
        model_output_stats = data[data['Name'] == model_prediction].select_dtypes(include=[float, int]).mean().to_dict()

        return jsonify({
            'user_input': user_data,
            'model_output_stats': model_output_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/othermodel')
def other_model():
    return render_template('othermodel.html', brands=unique_brands)

if __name__ == '__main__':
    app.run(debug=True)
