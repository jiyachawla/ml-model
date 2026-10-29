from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

# 1. FastAPI app initialize karein (bracket () lagana zaroori hai)
app = FastAPI()

# 2. Saved model ko variable me store karein
model = joblib.load("linearmodel.pkl")

# 3. Input data schema
class datascience(BaseModel):
    StudyHours: float

# 4. Root API endpoint
@app.get("/")
def test():
    return {"msg": "hello world"}

# 5. Prediction API endpoint
@app.post("/predict")
def predict_marks(data: datascience):
    prediction = model.predict([[data.StudyHours]])
    return {
        "StudyHours": data.StudyHours,
        "PredictedMarks": round(float(prediction[0]), 2)
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)