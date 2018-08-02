import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

def fit_and_predict(nome, modelo, dados):
    modelo.fit(dados['treino_dados'], dados['treino_marcacoes'])
    resultado = modelo.predict(dados['teste_dados'])
    acertos = resultado == dados['teste_marcacoes']

    total_acertos = sum(acertos)
    total_elementos = len(dados['teste_dados'])
    taxa_acertos = 100.0 * total_acertos/total_elementos
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_acertos)
    print(msg)
    return (taxa_acertos)

if __name__ == "__main__":

    df = pd.read_csv('buscas.csv')
    X_df = df[['home', 'busca', 'logado']]
    Y_df = df['comprou']
    Xdummies_df = pd.get_dummies(X_df).astype(int)
    Ydummies_df = Y_df
    X = Xdummies_df.values
    Y = Ydummies_df.values

    porcentagem_de_treino = 0.8
    porcentagem_de_teste = 0.1

    tamanho_de_treino = int(porcentagem_de_treino * len(Y))
    tamanho_de_teste = int(porcentagem_de_teste * len(Y))
    tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

    fim_de_teste = tamanho_de_treino + tamanho_de_teste

    dados = {}
    dados['treino_dados'] = X[:tamanho_de_treino]
    dados['treino_marcacoes'] = Y[:tamanho_de_treino]
    dados['teste_dados'] =  X[tamanho_de_treino:fim_de_teste]
    dados['teste_marcacoes'] = Y[tamanho_de_treino:fim_de_teste]
    dados['validacao_dados'] = X[fim_de_teste:]
    dados['validacao_marcacoes'] = Y[fim_de_teste:]

    modelo_abc = AdaBoostClassifier()
    modelo_mnb = MultinomialNB()

    resultado_mnb = fit_and_predict("MultinomialNB", modelo_mnb, dados)
    resultado_abc = fit_and_predict("AdaBoostClassifier", modelo_abc, dados)

    total_elementos = len(dados['teste_dados'])
    acertos_base = max(Counter(dados['teste_marcacoes']).itervalues())
    taxa_de_acerto_base = 100.0 * acertos_base / total_elementos

    print("Taxa de acerto base: %f" % taxa_de_acerto_base)
    print("Total de elementos analisados: %d" % total_elementos)

    if resultado_abc > resultado_mnb:
        vencedor = modelo_abc
    else:
        vencedor = modelo_mnb

    resultado = vencedor.predict(dados['validacao_dados'])
    acertos = resultado == dados['validacao_marcacoes']

    total_acertos = sum(acertos)

    taxa_acertos = 100.0 * total_acertos/total_elementos
    msg = "Taxa de acertos no mundo real eh {0}".format(taxa_acertos)
    print(msg)
