# Financial Document Classification with Machine Learning

This project is focused on building and deploying a machine learning model to classify financial documents based on their relevance. The project is structured to use both Google Colab for model training and PyCharm IDE for deployment purposes.

## Installation Instructions

### Prerequisites

- Python 3.7+
- Google Colab or a local environment with Jupyter support (for model training)
- PyCharm or any other Python IDE (for deployment)
- Git (optional, for version control)

### Steps to Set Up Locally

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model (Optional)**

   You can train the model using the provided notebook on Google Colab. The training process can be tracked using Weights and Biases (WandB). You can view the experiment tracking dashboard [here](https://wandb.ai/aljebraschool-university-muhammed-vi-polytechnic/financial_document_with_ml).

5. **Run the Application**

   You can run the API server locally using:

   ```bash
   uvicorn app:app --reload
   ```

   This will start the FastAPI server.

## Code Structure

The project is organized as follows:

```
my_fast_api_project/
├── models/
│   ├── best_model.joblib          # Serialized model file
│   ├── scaler.joblib              # Serialized scaler file
├── __init__.py
├── app.py                         # FastAPI application
├── gradio_interface.py            # Gradio interface for frontend
├── preprocessing.py               # Preprocessing script for data
├── requirements.txt               # Project dependencies
```

### Brief Description of Each File

- **app.py**: This file contains the FastAPI application, which includes endpoints for predicting financial document relevance using the trained model.
- **gradio_interface.py**: Provides a Gradio frontend interface for interacting with the model via a web interface.
- **preprocessing.py**: Handles all the data preprocessing tasks, including loading, scaling, and preparing data for model prediction.
- **models/best_model.joblib**: The best-trained machine learning model saved in a serialized format.
- **models/scaler.joblib**: The scaler used for normalizing input features before feeding them into the model.
- **requirements.txt**: Lists all the necessary Python packages and dependencies required to run the project.

## Summary

This project employs a robust machine learning pipeline to classify financial documents:

1. **Data Preprocessing**:
   - Merging feature and relevance data
   - Handling missing values and scaling features
   - Splitting the dataset into training and testing sets

2. **Model Training**:
   - Utilized four different models: Logistic Regression, Random Forest, Support Vector Machine, and XGBoost
   - Applied hyperparameter tuning using GridSearchCV and SMOTE for class imbalance
   - Tracked experiments and model performance using [Weights and Biases](https://wandb.ai/aljebraschool-university-muhammed-vi-polytechnic/financial_document_with_ml) for experiment tracking and visualization

3. **Model Deployment**:
   - Deployed the best model using FastAPI for API-based interaction
   - Integrated a Gradio interface for easy access to model predictions through a web interface

This project effectively demonstrates the end-to-end process of building, evaluating, and deploying a machine learning model for financial document classification.

