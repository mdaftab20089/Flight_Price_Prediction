

# Flight Price Prediction

## Overview
This project focuses on predicting flight prices using machine learning algorithms. The primary goal is to provide users with an accurate and efficient model that can estimate ticket prices based on various features, enabling better decision-making for travelers and airline companies. This project was developed collaboratively by Md Aftab, Sourav Sharma, and Tojum Tasar.

---

## Features
- **Data Analysis and Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables.
- **Exploratory Data Analysis (EDA)**: Insights into data distribution, correlations, and trends.
- **Feature Engineering**: Optimization of input features for better model performance.
- **Modeling**: Comparison of multiple machine learning models, including regression techniques.
- **Prediction and Evaluation**: Use of evaluation metrics like RMSE, MAE, and R^2 to validate model performance.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Jupyter Notebook
- **Machine Learning Techniques**:
  - Linear Regression
  - Ridge Regression
  - Polynomial Regression
  - Random Forest Regression (if applicable)

---

## Data
The dataset used for this project includes features such as:
- Airline
- Date of Journey
- Source and Destination
- Duration
- Total Stops
- Price (Target Variable)

### Data Preprocessing Steps:
1. Cleaning and handling missing values.
2. Converting categorical variables to numerical formats using techniques like one-hot encoding.
3. Normalizing or scaling numerical features.

---

## Project Structure
```
Flight_Price_Prediction/
|-- data/
|   |-- flight_data.csv        # Raw dataset
|   |-- processed_data.csv     # Cleaned and preprocessed dataset
|
|-- notebooks/
|   |-- EDA.ipynb              # Exploratory Data Analysis notebook
|   |-- Model_Building.ipynb   # Model training and evaluation notebook
|
|-- models/
|   |-- model.pkl              # Serialized model
|
|-- README.md                  # Project documentation
```

---

## Setup Instructions
### Prerequisites:
- Python >= 3.8
- Jupyter Notebook

### Installation Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/mdaftab20089/Flight_Price_Prediction
   ```
2. Navigate to the project directory:
   ```bash
   cd flight-price-prediction
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## Usage
1. Load the dataset provided in the `data/` directory.
2. Run the `EDA.ipynb` notebook to explore the data and gain insights.
3. Execute the `Model_Building.ipynb` notebook to train and evaluate the models.
4. Use the trained model for predictions by loading `model.pkl`.

---

---

## Future Scope
- Implement advanced machine learning algorithms like XGBoost or neural networks for improved accuracy.
- Develop a web or mobile interface for user-friendly predictions.
- Integrate real-time data sources for dynamic predictions.

---

## Contributors
- Md Aftab
- Sourav Sharma
- Tojum Tasar

Feel free to reach out for any queries or suggestions!

---


