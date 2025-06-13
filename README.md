# Crop Recommendation System

A machine learning-based web application that recommends the most suitable crop to grow based on soil and climate parameters. The system uses XGBoost algorithm trained on the Crop Recommendation Dataset to make accurate predictions.

## Features

- **User-friendly Interface**: Clean and intuitive web interface for inputting soil and climate parameters
- **Real-time Validation**: Input validation with proper ranges for each parameter
- **Accurate Predictions**: Machine learning model trained on cleaned and preprocessed data
- **Detailed Results**: Shows prediction along with input values for reference
- **Responsive Design**: Works on both desktop and mobile devices

## Technical Details

### Model
- Algorithm: XGBoost Classifier
- Features:
  - N (Nitrogen content in soil)
  - P (Phosphorus content in soil)
  - K (Potassium content in soil)
  - Temperature
  - Humidity
  - pH
  - Rainfall
- Number of Classes: 22 different crops
- Data Cleaning:
  - Removal of duplicate entries
  - Outlier detection and removal using Z-score method
  - Proper data type verification

### Technology Stack
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Machine Learning: XGBoost, scikit-learn
- Data Processing: pandas, numpy
- Model Persistence: joblib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crop-recommendation-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```bash
kaggle datasets download atharvaingle/crop-recommendation-dataset
unzip crop-recommendation-dataset.zip
```

4. Train the model:
```bash
python train_model.py
```

5. Run the application:
```bash
python app.py
```

6. Open your browser and go to `http://localhost:5000`

## Input Parameters

The system accepts the following parameters with their valid ranges:

| Parameter | Range | Unit |
|-----------|-------|------|
| Nitrogen (N) | 0-140 | kg/ha |
| Phosphorus (P) | 0-145 | kg/ha |
| Potassium (K) | 0-205 | kg/ha |
| Temperature | 8.0-44.0 | °C |
| Humidity | 14.0-100.0 | % |
| pH | 3.5-10.0 | - |
| Rainfall | 20.0-300.0 | mm |

## Project Structure

```
crop-recommendation-system/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── model.joblib           # Trained model
├── label_encoder.joblib   # Label encoder
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend template
└── README.md             # Project documentation
```

## Requirements

Create a `requirements.txt` file with the following dependencies:
```
flask==2.0.1
pandas==1.3.3
numpy==1.21.2
scikit-learn==0.24.2
xgboost==1.4.2
joblib==1.0.1
scipy==1.7.1
```

## Usage

1. Enter the soil and climate parameters in the web interface
2. Click "Get Crop Recommendation"
3. View the recommended crop along with the input values

## Model Performance

The model is trained on cleaned data with the following preprocessing steps:
- Removal of duplicate entries
- Outlier detection using Z-score method
- Proper data type verification

The model's performance metrics are displayed during training.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License



## Acknowledgments

- Dataset: [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- XGBoost documentation
- Flask documentation