from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open('titanic_model.pkl', 'rb'))

@app.get('/predict')
def predict(pclass: int, sex: int, age: float, sibsp: int, parch: int):
    data = pd.DataFrame([[pclass, sex, age, sibsp, parch]], columns=['pclass', 'sex', 'age', 'sibsp', 'parch'])

    prediction = model.predict(data)[0]

    if prediction ==1:
        return {"prediction": "Survived"}
    else:
        return({"prediction": "Not Survived"})
    