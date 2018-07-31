import csv
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('buscas.csv')
    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']
    Xdummies_df = pd.get_dummies(X_df)
    Ydummies_df = Y_df
    X = Xdummies_df.values
    Y = Ydummies_df.values
    print (X)
    print (Y)
