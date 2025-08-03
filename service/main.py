from fastapi import FastAPI, Request
import pickle
import numpy as np

# загрузите модель из файла выше
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# создаём приложение FastAPI
app = FastAPI(title="uplift")


@app.post("/predict")
async def predict(request: Request):
    # все данные передаются в json
    data = await request.json()

    # признаки лежат в features, в массиве
    # извлекаем и преобразуем признаки
    features = data.get("features")

    # получаем предсказания
    prediction = model.predict(np.array(features))

    return {"predict": prediction.tolist()}


if __name__ == "__main__":
    pass
# запустите сервис на хосте 0.0.0.0 и порту 5000
