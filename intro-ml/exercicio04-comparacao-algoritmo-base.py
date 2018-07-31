import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

if __name__ == "__main__":

    df = pd.read_csv('busca-sn.csv')
    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']
    Xdummies_df = pd.get_dummies(X_df).astype(int)
    Ydummies_df = Y_df
    X = Xdummies_df.values
    Y = Ydummies_df.values

    tamanho_teste = int(0.1 *len(Y))

    teste_dados =  X[-tamanho_teste:]
    teste_marcacoes = Y[-tamanho_teste:]
    modelo = joblib.load('ex03-persistencia-modelos.pk1')
    resultado = modelo.predict(teste_dados)
    acertos = resultado == teste_marcacoes

    total_acertos = sum(acertos)
    total_elementos = len(teste_dados)
    taxa_acertos = 100.0 * total_acertos/total_elementos

    acertos_base = max(Counter(teste_marcacoes).itervalues())
    taxa_de_acerto_base = 100.0 * acertos_base / total_elementos

    print("Taxa de acerto base: %f" % taxa_de_acerto_base)
    print("Taxa de acerto do algoritmo: %f" % taxa_acertos)
    print("Total de elementos analisados: %d" % total_elementos)
