import pandas as pd
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('Data.csv', delimiter=';')


data = pd.get_dummies(data, columns=['Zone', 'Type'], drop_first=True)

X = data.drop('Prix en dh', axis=1)
y = data['Prix en dh']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


def predict_price(zone, chambres, superficie, salon, type_maison):
    feature_names = X.columns
    input_data = pd.DataFrame([[0] * len(feature_names)], columns=feature_names)

   
    input_data.at[0, f'Zone_{zone}'] = 1
    input_data.at[0, 'Chambres'] = chambres
    input_data.at[0, 'Superficie'] = superficie
    input_data.at[0, 'Salon'] = salon
    input_data.at[0, f'Type_{type_maison}'] = 1

   
    price_prediction = model.predict(input_data[feature_names])[0]
    return price_prediction

print("Bienvenue dans le système d'estimation de prix immobilier.")
zone = input("Veuillez entrer la zone (A, B, C, D) : ")
chambres = int(input("Nombre de chambres : "))
superficie = int(input("Superficie en mètres carrés : "))
salon = int(input("Nombre de salons : "))
type_maison = input("Type de maison (Maison, Appartement, Studio, Villa) : ")


estimation_prix = predict_price(zone, chambres, superficie, salon, type_maison)
print(f"L'estimation du prix immobilier est d'environ {estimation_prix:.2f} DH.")
