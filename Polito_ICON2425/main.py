import warnings
import numpy as np
from customtkinter import set_default_color_theme

import model as m
from PIL import Image, ImageTk
import customtkinter as ctk

warnings.filterwarnings('ignore')

#Funzione per il pulsante di predizione
def predizione_prezzo():

    Label_Prediction.configure(text="")

    brand = Entry_Brand.get()
    index_brand = m.model.dataframe.columns.get_loc(F"brand_{brand}")

    model = Entry_Model.get()
    index_model = m.model.dataframe.columns.get_loc(F"model_{model}")

    model_year = int((Entry_Model_Year.get()))

    milage = float((Entry_Milage.get()))

    fuel_type = Entry_Fuel_Type.get()
    index_fuel_type = m.model.dataframe.columns.get_loc(F"fuel_type_{fuel_type}")

    engine = Entry_Engine.get()
    index_engine = m.model.dataframe.columns.get_loc(F"engine_{engine}")

    transmission = Entry_Transmission.get()
    index_transmission = m.model.dataframe.columns.get_loc(F"transmission_{transmission}")

    ext_col = Entry_External_Color.get()
    index_ext_col = m.model.dataframe.columns.get_loc(F"ext_col_{ext_col}")

    int_col = Entry_Internal_Color.get()
    index_int_col = m.model.dataframe.columns.get_loc(F"int_col_{int_col}")

    accident = Entry_Accident.get()

    clean_title = Entry_Clean_Title.get()

    #Predizione del prezzo
    num_features = m.model.dataframe.shape[1]  # Conta le colonne dopo il preprocessing
    sample = np.zeros((1, num_features))  # Usa il numero corretto di feature
    sample[0, :4] = np.array([model_year, milage, accident, clean_title]).reshape(1, -1)
    sample[0, index_brand] = 1
    sample[0, index_model] = 1
    sample[0, index_fuel_type] = 1
    sample[0, index_engine] = 1
    sample[0, index_transmission] = 1
    sample[0, index_ext_col] = 1
    sample[0, index_int_col] = 1

    sample = scaler.transform(sample.reshape(1, -1))

    prediction_model_selected = str(ComboBox_Prediction_Model.get())
    if(prediction_model_selected == 'Random Forest'):
        forest_modelPredict = forest_model.predict(sample)
        Label_Prediction.configure(text=("Il prezzo predetto è: € %.2f" %forest_modelPredict))
    elif (prediction_model_selected == 'SGD'):
        predicted_probabilities = SGD_model.predict_proba(sample).squeeze()

        # Ritrovo l'indice della probabilità maggiore
        index = np.argmax(predicted_probabilities) # Indice della probabilità maggiore
        probability = predicted_probabilities[index]  # Mi prendo la percentuale di probabilità in base all'indice qui sopra

        if(index == len(predicted_probabilities)-1):
            text = (F"Questo sample ha probabilità {probability*100:.2f}%\ndi rientrare nella fascia da € {(price_ranges[int(index)])[0]} in su.")
        else:
            text = (F"Questo sample ha probabilità {probability*100:.2f}%\ndi rientrare nella fascia € {price_ranges[int(index)]}.")
        Label_Prediction.configure(text=text)
    else:
        raise NotImplementedError(F"Scegli un modello valido. Hai scelto {prediction_model_selected}.")

def update_models(event):
    models = m.get_model_with_brand(Entry_Brand.get())
    Entry_Model.configure(values=models)
    Entry_Model.set(models[0])

price_ranges = [
    (0, 16000),
    (16000, 56000),
    (56000, 80000),
    (80000, 120000),
    (120000, 190000),
    (190000, float('inf'))
]

prices_x_train, prices_x_test, prices_y_train, prices_y_test, scaler  = m.crea_basedati(modelUsed="RandomForest")
prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD, scaler_SGD  = m.crea_basedati(modelUsed="SGD")

forest_model = m.modello(prices_x_train, prices_x_test, prices_y_train, prices_y_test)
SGD_model = m.modello2(prices_x_train_SGD, prices_x_test_SGD, prices_y_train_SGD, prices_y_test_SGD)

window = ctk.CTk()
ctk.set_appearance_mode("dark")
set_default_color_theme("blue")

# Impostiamo le dimensioni desiderate
window_width = 1100
window_height = 700

# Otteniamo la dimensione dello schermo
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calcoliamo la posizione per centrare la finestra
pos_x = (screen_width // 2) - (window_width // 2)
pos_y = (screen_height // 2) - (window_height // 2)

# Impostiamo la geometria con la posizione
window.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

window.title("Car Price Estimator")
window.resizable(False, False)
# Configura la colonna 0 per essere più piccola rispetto alle altre colonne
window.grid_columnconfigure(0, weight=1)  # Colonna 0 ha un peso più basso
window.grid_columnconfigure(1, weight=10)  # Colonna 1 avrà più spazio
window.grid_columnconfigure(2, weight=10)  # Colonna 2 avrà lo stesso spazio della colonna 1

Label_Titolo = ctk.CTkLabel(window, text="", font=("Helvetica", 20, "bold"))
Label_Titolo.grid(row=0,column=0, columnspan=3, padx=10)

image_path = "image.jpg"  # Cambia con il percorso dell'immagine
img = Image.open(image_path)  # Carica l'immagine
img = img.resize((600, 600))  # Ridimensiona l'immagine (puoi cambiare le dimensioni)
img_tk = ImageTk.PhotoImage(img)  # Converte l'immagine in un formato compatibile con Tkinter

Label_Image = ctk.CTkLabel(window, image=img_tk, text="")
Label_Image.grid(row=1, column=2, rowspan=12, padx=20, sticky="ns")

# Creo l'etichetta e la combobox per il valore "Brand"
Label_Brand = ctk.CTkLabel(window, text="Casa costruttrice: ", font=("Helvetica", 16))
Label_Brand.grid(row=1, column=0, padx=10, sticky="e")
Entry_Brand = ctk.CTkComboBox(window, values=m.get_brand(), state='readonly', command=update_models)
Entry_Brand.grid(row=1, column=1, padx=10, sticky="ew")
Entry_Brand.set(m.get_brand()[0])  # Seleziona il primo brand

# Creo l'etichetta e la combobox per il valore "Model"
Label_Model = ctk.CTkLabel(window, text="Modello: ", font=("Helvetica", 16))
Label_Model.grid(row=2, column=0, padx=10, sticky="e")
Entry_Model = ctk.CTkComboBox(window, state='readonly')
Entry_Model.grid(row=2, column=1, padx=10, sticky="ew")

# Associo la callback al cambio del brand e aggiorno subito la lista dei modelli
Entry_Brand.bind("<<ComboboxSelected>>", update_models)
update_models(None)

#Creo l'etichetta e la combobox per il valore "Model Year"
Label_Model_Year = ctk.CTkLabel(window, text = "Anno: ", font=("Helvetica", 16))
Label_Model_Year.grid(row=3, column=0, padx=10, sticky="e")
Entry_Model_Year = ctk.CTkComboBox(window, values=[str(year) for year in m.get_model_year()], state='readonly')
Entry_Model_Year.grid(row=3, column=1, padx=10, sticky="ew")
Entry_Model_Year.set(m.get_model_year()[0])

#Creo l'etichetta e la combobox per il valore "Milage"
Label_Milage = ctk.CTkLabel(window, text = "Km percorsi: ", font=("Helvetica", 16))
Label_Milage.grid(row=4, column=0, padx=10, sticky="e")
Entry_Milage = ctk.CTkComboBox(window, values=[str(milage) for milage in m.get_milage()], state = 'readonly')
Entry_Milage.grid(row=4, column=1, padx=10, sticky="ew")
Entry_Milage.set(m.get_milage()[0])

#Creo l'etichetta e la combobox per il valore "Fuel Type"
Label_Fuel_Type = ctk.CTkLabel(window, text = "Tipo di carburante: ", font=("Helvetica", 16))
Label_Fuel_Type.grid(row=5, column=0, padx=10, sticky="e")
Entry_Fuel_Type = ctk.CTkComboBox(window, values = m.get_fuel_type(), state = 'readonly')
Entry_Fuel_Type.grid(row=5, column=1, padx=10, sticky="ew")
Entry_Fuel_Type.set(m.get_fuel_type()[0])

#Creo l'etichetta e la combobox per il valore "Engine"
Label_Engine = ctk.CTkLabel(window, text = "Motore: ", font=("Helvetica", 16))
Label_Engine.grid(row=6, column=0, padx=10, sticky="e")
Entry_Engine = ctk.CTkComboBox(window, values = m.get_engine(), state = 'readonly')
Entry_Engine.grid(row=6, column=1, padx=10, sticky="ew")
Entry_Engine.set(m.get_engine()[0])

#Creo l'etichetta e la combobox per il valore "Transmission"
Label_Transmission = ctk.CTkLabel(window, text = "Trasmissione: ", font=("Helvetica", 16))
Label_Transmission.grid(row=7, column=0, padx=10, sticky="e")
Entry_Transmission = ctk.CTkComboBox(window, values = m.get_transmission(), state = 'readonly')
Entry_Transmission.grid(row=7, column=1, padx=10, sticky="ew")
Entry_Transmission.set(m.get_transmission()[0])

#Creo l'etichetta e la combobox per il valore "External Color"
Label_External_Color = ctk.CTkLabel(window, text = "Colore esterni: ", font=("Helvetica", 16))
Label_External_Color.grid(row=8, column=0, padx=10, sticky="e")
Entry_External_Color = ctk.CTkComboBox(window, values = m.get_ext_col(), state = 'readonly')
Entry_External_Color.grid(row=8, column=1, padx=10, sticky="ew")
Entry_External_Color.set(m.get_ext_col()[0])

#Creo l'etichetta e la combobox per il valore "Internal Color"
Label_Internal_Color = ctk.CTkLabel(window, text = "Colore interni: ", font=("Helvetica", 16))
Label_Internal_Color.grid(row=9, column=0, padx=10, sticky="e")
Entry_Internal_Color = ctk.CTkComboBox(window, values = m.get_int_col(), state = 'readonly')
Entry_Internal_Color.grid(row=9, column=1, padx=10, sticky="ew")
Entry_Internal_Color.set(m.get_int_col()[0])

#Creo l'etichetta e lo spinbox per il valore "Accident"
Label_Accident = ctk.CTkLabel(window, text = "Incidenti: ", font=("Helvetica", 16))
Label_Accident.grid(row=10, column=0, padx=10, sticky="e")
Entry_Accident = ctk.CTkComboBox(window, values=["0", "1"], state='readonly')
Entry_Accident.grid(row=10, column=1, padx=10, sticky="ew")
Entry_Accident.set("0")  # Imposta il valore di default

#Creo l'etichetta e lo spinbox per il valore "Clean Title"
Label_Clean_Title = ctk.CTkLabel(window, text = "Titolo pulito: ", font=("Helvetica", 16))
Label_Clean_Title.grid(row=11, column=0, padx=10, sticky="e")
Entry_Clean_Title = ctk.CTkComboBox(window, values=["0", "1"], state='readonly')
Entry_Clean_Title.grid(row=11, column=1, padx=10, sticky="ew")
Entry_Clean_Title.set("1")  # Imposta il valore di default

# ComboBox per il modello di predizione
Label_Prediction_Model = ctk.CTkLabel(window, text = "Modello di predizione: ", font=("Helvetica", 16))
Label_Prediction_Model.grid(row=12, column=0, padx=10, sticky="e")
ComboBox_Prediction_Model = ctk.CTkComboBox(window, values = m.get_prediction_model(), state = 'readonly')
ComboBox_Prediction_Model.grid(row=12, column=1, padx=10, sticky="ew")
ComboBox_Prediction_Model.set(m.get_prediction_model()[0])

# Label per la predizione
Label_Prediction = ctk.CTkLabel(window, text="Qui viene mostrato il risultato della predizione", font=("Helvetica", 16), bg_color="gray",  # Colore di sfondo
                                text_color="white",  # Colore del testo
                                corner_radius=10)
Label_Prediction.grid(row=14, column=1, padx=10, sticky="ew")

#Creo il pulsante per il passaggio dei valori dell'utente
getValue_button = ctk.CTkButton(window, text="Avvia Predizione", command=predizione_prezzo)
getValue_button.grid(row=13, column=1, padx=10, pady=20)

if __name__ == "__main__":
    window.mainloop()