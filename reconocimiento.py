from deepface import DeepFace

# Imagen a identificar
img = "prueba.jpeg"

# Base de datos de rostros
db = "fotos/"

# Buscar coincidencias
resultados = DeepFace.find(img_path=img, db_path=db, model_name="ArcFace",     enforce_detection=False )

print(resultados)
