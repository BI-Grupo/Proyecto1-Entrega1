from pydantic import BaseModel

class DataModel(BaseModel):

    Textos_espanol: str
    

#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return ["Textos_espanol"]
