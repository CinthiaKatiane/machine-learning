import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
if __name__ == "__main__":

    df = pd.read_csv('buscas.csv')
    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']
    Xdummies_df = pd.get_dummies(X_df).astype(int)
    Ydummies_df = Y_df
    X = Xdummies_df.values
    Y = Ydummies_df.values

    tamanho_treino = int(0.9 * len(Y))
    tamanho_teste = len(Y) - tamanho_treino

    teste_dados =  X[-tamanho_teste:]
    teste_marcacoes = Y[-tamanho_teste:]

    modelo = joblib.load('ex03-persistencia-modelos.pk1')

    resultado = modelo.predict(teste_dados)

    diferencas = resultado - teste_marcacoes
    acertos = [d for d in diferencas if d == 0]
    total_acertos = len(acertos)
    total_elementos = len(teste_dados)

    taxa_acertos = 100.0 * total_acertos/total_elementos

    print (taxa_acertos)
    print (total_elementos)
