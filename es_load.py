import joblib
model = joblib.load("mymlp.joblib")

feature_scaler = joblib.load("./feature_scaler.joblib")
target_encoder = joblib.load("./target_encoder.joblib")

pinguino_test = [[38, 15.6, 205.0, 3780.0]] # mondo feature reali

specie_num = model.predict(feature_scaler.transform(pinguino_test))

print(target_encoder.inverse_transform([specie_num]))
