import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Template setup
templates = Jinja2Templates(directory="templates")

# Input schema for API
class InputData(BaseModel):
    TV: float
    Radio: float
    Newspaper: float

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request, TV: float = Form(...), Radio: float = Form(...), Newspaper: float = Form(...)):
    features = np.array([[TV, Radio, Newspaper]])
    prediction = model.predict(features)[0]
    return templates.TemplateResponse("index.html", {"request": request, "prediction": round(prediction, 2)})

@app.post("/predict/")
def predict(data: InputData):
    features = np.array([[data.TV, data.Radio, data.Newspaper]])
    prediction = model.predict(features)[0]
    return {"predicted_sales": prediction}
