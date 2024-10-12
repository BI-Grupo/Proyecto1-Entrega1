from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from joblib import load, dump
import pandas as pd
from Pipeline import pipeline
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from DataModel import DataModel
from fastapi.responses import StreamingResponse
from io import BytesIO

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def make_predictions(file: UploadFile):

    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))

        model = load("modelo.joblib")
        df["prediction"] = model.predict(df)
        
        # Crear un archivo Excel con las predicciones
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        # Devolver el archivo Excel con las predicciones
        return StreamingResponse(output, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 headers={"Content-Disposition": "attachment; filename=predicciones.xlsx"})
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/retrain")
async def retrain_model(file: UploadFile):
    # Leer el archivo Excel recibido
    try:
        contents = await file.read()
        new_data = pd.read_excel(BytesIO(contents))
        
    except Exception as e:
        return {"error": f"Error leyendo el archivo: {str(e)}"}
    
    # Verificar que las columnas requeridas estén presentes
    if 'Textos_espanol' not in new_data.columns or 'sdg' not in new_data.columns:
        return {"error": "El archivo debe contener las columnas 'Textos_espanol' y 'sdg'."}
    
    # Eliminar filas con valores nulos en 'Textos_espanol' o 'sdg'
    new_data = new_data.dropna(subset=['Textos_espanol', 'sdg'])
    
    # Cargar el archivo base existente
    base_data_path = './data/registros.xlsx'
    if not os.path.exists(base_data_path):
        return {"error": "El archivo base 'registros.xlsx' no existe."}

    base_data = pd.read_excel(base_data_path)

    # Concatenar el nuevo dataset con el dataset base
    updated_data = pd.concat([base_data, new_data], ignore_index=True)

    # Guardar el dataset actualizado (sobreescribe el archivo base)
    updated_data.to_excel(base_data_path, index=False)

    # Separar características y etiquetas
    X = updated_data['Textos_espanol']
    y = updated_data['sdg']

    # Reentrenar el modelo
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        pipeline.fit(X_train, y_train)

        # Guardar el nuevo modelo entrenado
        dump(pipeline, 'modelo.joblib')
    except Exception as e:
        return {"error": f"Error durante el reentrenamiento: {str(e)}"}

    # Calcular métricas de evaluación
    y_pred = pipeline.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        "message": "Reentrenamiento exitoso.",
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
