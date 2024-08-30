from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Union, List
from typing import List, Optional
import pandas as pd
import io
import joblib
from preprocessing import preprocess_documents  # Import preprocessing function

app = FastAPI()

# Load model and scaler
model = joblib.load("models/best_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# Define input schema
class PredictionInput(BaseModel):
    DocumentID: str
    Title: str
    RegulatorId: str
    SourceLanguage: str
    DocumentTypeId: str
    CombinedScore: float
    Content: str

# Define output schema
class PredictionOutput(BaseModel):
    DocumentID: str
    Prediction: str
    Confidence: float
    Explanation: Union[List[dict], str]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))

        preprocessed_df = preprocess_documents(df)

        features = preprocessed_df[['RegulatorScore', 'DocTypeScore', 'KeywordScore', 'LanguageScore']] # 'CombinedScore'
        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)
        confidences = model.predict_proba(scaled_features).max(axis=1)

        output_df = pd.DataFrame({
            'DocumentID': preprocessed_df['DocumentID'],
            'Prediction': ["Relevant" if pred == 1 else "Irrelevant" for pred in predictions],
            'Confidence': confidences,
            'Explanation': features.apply(lambda row: ', '.join(f"{col}: {value:.2f}" for col, value in row.items()), axis=1)
        })

        output = io.StringIO()
        output_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(iter([output.getvalue()]),
                                 media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=predictions.csv"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

