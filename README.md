# Financial Document Classification with Machine Learning

This project focuses on classifying financial regulations documents using various machine learning models. The goal is to filter relevant documents based on certain criteria and provide accurate predictions using models like Logistic Regression, Random Forest, Support Vector Machine, and XGBoost. The project also leverages the Weights & Biases (WandB) platform for experiment tracking and model management.

## Installation Instructions

To set up this project on your local machine, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/financial_document_classification.git
   cd financial_document_classification
   ```

2. **Set Up the Python Environment:**
   Ensure you have Python 3.8 or later installed. You can create a virtual environment and install the required packages using:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Model Training (Optional):**
   The model was trained using Google Colab. You can view the training process or rerun it by accessing the following Colab notebook:
   [Google Colab Notebook](https://colab.research.google.com/drive/1akHZK6wGip3DKjcmkbXnj7BARsf9pU9B?usp=sharing)

   Alternatively, you can use the pre-trained model included in this repository to skip the training step.

## Running the API

The project includes a FastAPI application for serving model predictions. Follow these steps to run the API:

1. **Start the FastAPI Server:**
   In your terminal, navigate to the project directory and run:
   ```bash
   uvicorn app:app --reload
   ```
   This command will start a local server at `http://127.0.0.1:8000`.

2. **Test the API:**
   Once the server is running, you can test the API by sending a POST request to the `/predict` endpoint. You can use tools like `curl`, Postman, or simply navigate to the Swagger UI at `http://127.0.0.1:8000/docs` to interact with the API.

   The API expects a CSV file as input, and it will return a CSV file with the predictions. An example input file is provided in the repository.

## Running the Gradio Interface

Gradio provides a user-friendly interface to interact with the model. Follow these steps to launch the Gradio interface:

1. **Start the Gradio Interface:**
   Run the following command in your terminal:
   ```bash
   python gradio_interface.py
   ```

2. **Interact with the Model:**
   After starting the interface, Gradio will provide you with a link. Open the link in your web browser to interact with the model. You can upload documents and see the predictions directly.

## Code Structure

The codebase is organized as follows:

```plaintext
my_fast_api_project/
│
├── models/
│   ├── best_model.joblib         # Pre-trained model
│   ├── scaler.joblib             # Scaler used for feature normalization
│
├── __init__.py                   # Package initializer
├── app.py                        # FastAPI app with endpoints for predictions
├── gradio_interface.py           # Gradio interface for interacting with the model
├── preprocessing.py              # Preprocessing steps for input data
├── requirements.txt              # Dependencies required to run the project
```

- **`app.py`:** Contains the FastAPI code for serving model predictions via an API.
- **`gradio_interface.py`:** Implements a Gradio interface for a more interactive model testing experience.
- **`preprocessing.py`:** Handles data preprocessing tasks such as scaling and feature engineering.

## Machine Learning Pipeline Overview

The machine learning pipeline follows these steps:

1. **Data Preprocessing:**
   - Data is cleaned and preprocessed using scaling, encoding, and feature extraction techniques.
   - TF-IDF vectors are generated from text data to represent the documents numerically.

2. **Model Training:**
   - Multiple models (Logistic Regression, Random Forest, Support Vector Machine, XGBoost) were trained and evaluated using GridSearchCV for hyperparameter tuning.
   - The training process was tracked using Weights & Biases for experiment management. You can view the training logs and model performance on WandB:
     [WandB Experiment Link](https://wandb.ai/aljebraschool-university-muhammed-vi-polytechnic/financial_document_with_ml)

3. **Model Evaluation:**
   - The models were evaluated based on accuracy, precision, recall, and F1-score to ensure robust performance on the classification task.
   - SHAP (SHapley Additive exPlanations) was used for model explainability, highlighting which features were most influential in the predictions.

4. **Deployment:**
   - The trained model is served using a FastAPI application. Additionally, a Gradio interface is provided for easier interaction with the model for testing purposes.

## Additional Resources

For more detailed steps and insights into the model training, please refer to the [Google Colab Notebook](https://colab.research.google.com/drive/1akHZK6wGip3DKjcmkbXnj7BARsf9pU9B?usp=sharing).
