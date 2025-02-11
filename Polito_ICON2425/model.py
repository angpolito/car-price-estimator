import csv
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
from sklearn.calibration import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

warnings.filterwarnings('ignore')
indexState = 497 # R2: 0.7831
class model:
    dataframe = None
    dataframe_UI = None
    cars = None

    brand = None
    model = None
    model_year = None
    milage = None
    fuel_type = None
    engine = None
    transmission = None
    ext_col = None
    int_col = None
    accident = None
    clean_title = None

    prediction_model = None

    features = ['brand', 'model', 'model_year', 'milage', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'price']

# Function to map house prices to price range categories
def map_to_price_range(price):
    price_ranges = [
        (0, 16000),
        (16000, 56000),
        (56000, 80000),
        (80000, 120000),
        (120000, 190000),
        (190000, float('inf'))
    ]
    for idx, (lower, upper) in enumerate(price_ranges):
        if lower <= price < upper:
            return idx
    return len(price_ranges)

def crea_basedati(modelUsed='None'):

    # Leggi i dati dal file csv
    cars_data = pd.read_csv("used_cars.csv", index_col=False)

    data = pd.DataFrame(cars_data)

    # Conversione dei tipi dei dati
    data['brand'] = data['brand'].astype('string')
    data['model'] = data['model'].astype('string')
    data['model_year'] = data['model_year'].astype('int64')

    data['milage'] = data['milage'].str.replace('"', '', regex=True)  # Rimuove i doppi apici
    data['milage'] = data['milage'].str.replace(' mi.', '', regex=True)  # Rimuove " mi."
    data['milage'] = data['milage'].str.replace(',', '', regex=True)  # Rimuove la virgola
    data['milage'] = data['milage'].astype('float32')

    data['fuel_type'] = data['fuel_type'].astype('string')
    data['engine'] = data['engine'].astype('string')
    data['transmission'] = data['transmission'].astype('string')
    data['ext_col'] = data['ext_col'].astype('string')
    data['int_col'] = data['int_col'].astype('string')
    data['accident'] = data['accident'].astype('string')

    # Conversione della colonna 'accident' in valori binari
    data['accident'] = data['accident'].map({
        'At least 1 accident or damage reported': 1,
        'None reported': 0
    })

    data['clean_title'] = data['clean_title'].astype('string')

    # Conversione della colonna 'clean_title' in valori binari
    data['clean_title'] = data['clean_title'].map({
        'Yes': 1,
        'No': 0
    })

    data['price'] = data['price'].str.replace(r'[\$,"]', '', regex=True) # Rimuove doppi apici, simbolo del dollaro e virgole
    data['price'] = data['price'].astype('float32')

    # Elimina i duplicati
    data.drop_duplicates(inplace=True)

    # Rimuovi i Nan, -, --
    # Usa regex per rimuovere trattini singoli o doppi che sono l'intero contenuto della stringa, e 'not supported'
    # Funzione di pulizia per ogni colonna

    toClean = ['fuel_type', 'engine', 'transmission', 'ext_col', 'int_col']

    for col in toClean:
        data[col] = data[col].apply(
            lambda x: np.nan if isinstance(x, str) and (re.match(r"^[-–]+$", x) or x.lower() == "not supported") else x)

    # Eliminazione dei Nan
    data.dropna(inplace=True)

    # Conversione miglia in kilometri
    scaleKm = 1.6093
    data['milage'] = (data['milage'] * scaleKm).round(2)

    # Creazione di una copia di data per lasciare data integro e operare su cars
    cars = data.copy()

    # Creazione di liste ordine di elementi unici
    model.dataframe_UI = cars
    model.brand = sorted(list(cars.brand.unique()))
    model.model = sorted(list(cars.model.unique()))
    model.model_year = sorted(list(cars.model_year.unique()))
    model.model_year.reverse()
    model.milage = sorted(list(cars.milage.unique()))
    model.fuel_type = sorted(list(cars.fuel_type.unique()))
    model.engine = sorted(list(cars.engine.unique()))
    model.transmission = sorted(list(cars.transmission.unique()))
    model.ext_col = sorted(list(cars.ext_col.unique()))
    model.int_col = sorted(list(cars.int_col.unique()))
    model.accident = sorted(list(cars.accident.unique()))
    model.clean_title = sorted(list(cars.clean_title.unique()))

    model.prediction_model = ['Random Forest', "SGD"]
    
    # Processing dei dati non numerici con i metodi get_dummies() e LabelEncoder()
    car = pd.get_dummies(data, columns=['brand'], prefix='brand')
    car = pd.get_dummies(car, columns=['model'], prefix='model')
    car = pd.get_dummies(car, columns=['fuel_type'], prefix='fuel_type')
    car = pd.get_dummies(car, columns=['engine'], prefix='engine')
    car = pd.get_dummies(car, columns=['transmission'], prefix='transmission')
    car = pd.get_dummies(car, columns=['ext_col'], prefix='ext_col')
    car = pd.get_dummies(car, columns=['int_col'], prefix='int_col')

    le = LabelEncoder()
    car['accident'] = le.fit_transform(car['accident'])
    car['clean_title'] = le.fit_transform(car['clean_title'])

    columns = car.columns
    columns = columns.drop('price')

    # Processo per rendere i dati degli scalari
    scaler = preprocessing.MinMaxScaler(feature_range = (0, 1))

    normal = pd.DataFrame(scaler.fit_transform(car.loc[:, car.columns!='price']), columns = columns)
    normal = normal.reset_index(drop=True, inplace=False)

    model.dataframe = normal.copy()

    prices_x = normal
    # ---------------- #
    if(modelUsed == 'RandomForest'):
        prices_y = pd.DataFrame(car["price"]) # Prezzi normali
    elif (modelUsed == 'SGD'):
        prices_y = pd.DataFrame(car["price"])
        # Applico la funzione per creare la variabile target (categorie fascia di prezzo)
        prices_y['price'] = prices_y['price'].apply(map_to_price_range)
    else:
        raise NotImplementedError(F"Parametro modelUsed non corretto. Hai usato: '{modelUsed}'.")

    # Apprendimento non supervisionato con clustering
    clusters = DBSCAN(eps=2.5, min_samples=15).fit(prices_x)
    prices_x["noise"] = clusters.labels_
    prices_y["noise"] = clusters.labels_

    prices_x = prices_x[prices_x.noise>-1]
    prices_y = prices_y[prices_y.noise>-1]
    prices_x.drop('noise', inplace = True, axis=1)
    prices_y.drop('noise', inplace = True, axis=1)

    #Allenamento e test degli split
    np.random.seed(indexState)
    prices_x_train, prices_x_test, prices_y_train, prices_y_test = train_test_split(prices_x, prices_y, test_size=0.2)

    prices_x_train = prices_x_train.to_numpy()
    prices_x_test = prices_x_test.to_numpy()
    prices_y_train = prices_y_train.to_numpy()
    prices_y_test = prices_y_test.to_numpy()

    """
    print('Training set size: %d' %len(prices_x_train))
    print('Test set size: %d' %len(prices_x_test))
    print('----------------------------------------------')
    print(F'Shape of X values for Training set: {prices_x_train.shape}')
    print(F'Shape of Y values for Training set: {prices_y_train.shape}')
    print('----------------------------------------------')
    """

    return prices_x_train, prices_x_test, prices_y_train, prices_y_test, scaler

#Funzione per plottare i grafici
def plot(predictions, x, y, n_feature):
    fig = plt.figure(dpi=125)
    fig.set_figwidth(10)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.elev = 20
    ax.azim = 20
    ax.scatter3D(x[:,n_feature], y, edgecolors='blue', alpha=0.5)
    ax.scatter3D(x[:,n_feature], predictions, 0.00, linewidth=0.5, edgecolors='red', alpha=0.7)

    # ==============
    # Second subplot
    # ==============
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.elev = 20
    ax.azim = 20

    ax.scatter3D(x[:,n_feature], y, edgecolors='blue', alpha=0.5)
    ax.scatter3D(x[:,n_feature], predictions, 0.02, linewidth=0.5, edgecolors='red', alpha=0.7)

    title = ""

    plt.suptitle(title)
    plt.show()

def modello2(prices_x_train, prices_x_test, prices_y_train, prices_y_test):

    #Crea e allena il modello SGD
    SGD_model = SGDClassifier(loss="log_loss", n_jobs=-1, alpha=0.0001, random_state=indexState)

    """
    # ========== Grid Search  ========== #
    xConcatenato = np.concatenate((prices_x_train, prices_x_test), axis=0)
    yConcatenato = np.concatenate((prices_y_train, prices_y_test), axis=0)

    cv = KFold(n_splits=10, shuffle=True, random_state=indexState)
    listaAccuracy = []
    i = 0
    print("-- Grid Search per SGD --\n")

    for tr_idx, test_idx in cv.split(xConcatenato, yConcatenato):
        i += 1
        correctGrid = 0
        X_train, X_test = xConcatenato[tr_idx], xConcatenato[test_idx]
        y_train, y_test = yConcatenato[tr_idx], yConcatenato[test_idx]

        grid_search_regression = GridSearchCV(SGD_model,
                                              {
                                                  'alpha': np.arange(0.0001, 0.0006, 0.0001),
                                                  'fit_intercept': [True, False],
                                              }, cv=cv, scoring="accuracy", verbose=1, n_jobs=-1
                                              )

        grid_search_regression.fit(X_train, np.squeeze(y_train).ravel())

        gridSearchPred = grid_search_regression.predict_proba(X_test)
        gridSearchPred = np.argmax(gridSearchPred, axis=1)
        correctGrid += (gridSearchPred == np.squeeze(y_test)).sum()
        accuracyPerFold = (correctGrid / y_test.size) * 100
        listaAccuracy.append(accuracyPerFold)
        print(F"Split n.{i} | Best hyper-parameters per SGD: {grid_search_regression.best_params_}")
        print(f"Accuracy split n.{i}: {accuracyPerFold:.2f}%")

    split_numbers = np.arange(1, 10 + 1)
    plt.plot(split_numbers, listaAccuracy, label='Accuracies')
    plt.xlabel('Split')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Grid Search - SGD')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    # ========== SGD normale ========== #
    correct = 0
    SGD_model.fit(prices_x_train, prices_y_train)
    y_pred = SGD_model.predict_proba(prices_x_test)
    y_pred_probabilities = y_pred
    y_pred = np.argmax(y_pred, axis=1)
    correct += (y_pred == np.squeeze(prices_y_test)).sum()

    # print(f"Accuracy SGD: {(correct/prices_y_test.size)*100:.2f}%")

    # disp = printConfusionMatrix(prices_y_test, y_pred, SGD_model, 'SGD').plot()
    # plt.show()
    return SGD_model

def modello(prices_x_train, prices_x_test, prices_y_train, prices_y_test):

    # Apprendimento supervisionato
    forest_model = RandomForestRegressor(n_jobs=-1, random_state=indexState)

    """
    # # ========== Grid Search ========== #
    xConcatenato = np.concatenate((prices_x_train, prices_x_test), axis=0)
    yConcatenato = np.concatenate((prices_y_train, prices_y_test), axis=0)

    cv = KFold(n_splits=10, shuffle=True, random_state=indexState)
    listaR2Scores = []
    i = 0
    print("-- Grid Search per forestmodel --\n")
    for tr_idx, test_idx in cv.split(xConcatenato, yConcatenato):
        i += 1
        X_train, X_test = xConcatenato[tr_idx], xConcatenato[test_idx]
        y_train, y_test = yConcatenato[tr_idx], yConcatenato[test_idx]

        grid_search_regression = GridSearchCV(forest_model,
                                {
                                'n_estimators':np.arange(100,200,10),
                                'criterion': ['mae', 'friedman_mse'],
                                'bootstrap': [True, False],
                                'warm_start': [True, False],
                                }, cv=cv, scoring="neg_root_mean_squared_error", verbose=10, n_jobs=-1
                                )

        grid_search_regression.fit(X_train, np.squeeze(y_train).ravel())

        gridSearchPred = grid_search_regression.predict(X_test)
        r2score = r2_score(y_test, gridSearchPred)

        listaR2Scores.append(r2score)
        print(F"Split n.{i} | Best hyper-parameters per Forest Model: {grid_search_regression.best_params_}")
        print(f"R2 score split n.{i}: {r2score:.2f}%")

    split_numbers = np.arange(1, 10 + 1)
    plt.plot(split_numbers, listaR2Scores, label='Scores')
    plt.xlabel('Split')
    plt.ylabel('R2 Score')
    plt.title('R2 Scores Grid Search - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("-----------------------------------------------------------------------------------")
    """

    # ========== RandomForest normale ========== #
    # Addestramento finale del modello sul training set completo
    forest_model.fit(prices_x_train, prices_y_train)

    # Valutazione del modello sul test set
    y_pred = forest_model.predict(prices_x_test)

    """
    #Controllo dei punteggi
    print(f"MAE RandomForest normale sul test set: {mean_absolute_error(prices_y_test, y_pred):.2f}")
    print(f"MSE RandomForest normale sul test set: {mean_squared_error(prices_y_test, y_pred):.2f}")
    print(f"R2 Score RandomForest normale sul test set: {r2_score(prices_y_test, y_pred):.2f}")
    print("-----------------------------------------------------------------------------------")
    """

    #Predizione del modello
    return forest_model

def printConfusionMatrix(targets, predictions, classifierConf, label):
    class_names = np.array(["0","1","2","3","4","5"])

    print(F"Classification report for {label}:")
    print(classification_report(targets, predictions, labels=[0,1,2,3,4,5], target_names=class_names, digits=3, zero_division="warn"))
    print("============================================")

    cm = confusion_matrix(targets, predictions, labels=classifierConf.classes_)

    return ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifierConf.classes_)

def get_brand():
    return model.brand

def get_prediction_model():
    return model.prediction_model

def get_model():
    return model.model

def get_model_year():
    return model.model_year

def get_milage():
    return model.milage

def get_fuel_type():
    return model.fuel_type

def get_engine():
    return model.engine

def get_transmission():
    return model.transmission

def get_ext_col():
    return model.ext_col

def get_int_col():
    return model.int_col

def get_accident():
    return model.accident

def get_clean_title():
    return model.clean_title

def get_model_with_brand(brand):
    # Seleziona le righe in cui la colonna 'brand' corrisponde al brand richiesto
    df_brand = model.dataframe_UI[model.dataframe_UI['brand'] == brand]
    # Ottieni i valori unici della colonna 'model'
    models = df_brand['model'].unique()
    # Ordina i modelli in ordine alfabetico
    lista = sorted(models)
    return lista