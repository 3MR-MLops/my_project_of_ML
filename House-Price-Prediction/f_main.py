from fastapi import FastAPI
import numpy as np
import tensorflow as tf
app = FastAPI()

model = tf.keras.models.load_model('PR_One.keras')

@app.get("/predict")
def predict_price(area: float, rooms: int, age: int, has_pool: int):
    input_data = np.array([[area, rooms, age, has_pool]])
    prediction = model.predict(input_data)
    return {"predicted_price": f"{prediction[0][0]:.2f} K"}

