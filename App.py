import tkinter as tk
from tkinter import Label, Entry, Button
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

def on_predict_button():
    zone = zone_entry.get()
    chambres = int(chambres_entry.get())
    superficie = int(superficie_entry.get())
    salon = int(salon_entry.get())
    type_maison = type_maison_entry.get()
    estimation_prix = predict_price(zone, chambres, superficie, salon, type_maison)
    result_label.config(text=f"L'estimation du prix immobilier est d'environ {estimation_prix:.2f} DH.")

app = tk.Tk()
app.title("Estimation de Prix Immobilier")


Label(app, text="Zone (A, B, C, D): ").grid(row=0, column=0)
zone_entry = Entry(app)
zone_entry.grid(row=0, column=1)

Label(app, text="Nombre de chambres: ").grid(row=1, column=0)
chambres_entry = Entry(app)
chambres_entry.grid(row=1, column=1)

Label(app, text="Superficie en mètres carrés: ").grid(row=2, column=0)
superficie_entry = Entry(app)
superficie_entry.grid(row=2, column=1)

Label(app, text="Nombre de salons: ").grid(row=3, column=0)
salon_entry = Entry(app)
salon_entry.grid(row=3, column=1)

Label(app, text="Type de maison (Maison, Appartement, Studio, Villa): ").grid(row=4, column=0)
type_maison_entry = Entry(app)
type_maison_entry.grid(row=4, column=1)


predict_button = Button(app, text="Prédire le Prix", command=on_predict_button)
predict_button.grid(row=5, column=0, columnspan=2)


result_label = Label(app, text="")
result_label.grid(row=6, column=0, columnspan=2)

app.mainloop()
