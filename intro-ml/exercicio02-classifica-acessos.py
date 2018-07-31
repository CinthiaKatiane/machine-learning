import csv

from sklearn.naive_bayes import MultinomialNB

def carregar_acessos():
    X = []
    Y = []

    arquivo = open('acesso.csv','r')
    leitor = csv.reader(arquivo)
    leitor.next()
    for home, como_funciona, contato, comprou  in leitor:
        X.append([int(home), int(como_funciona), int(contato)])
        Y.append(int(comprou))
    return X, Y

if __name__ == '__main__':

    X,Y = carregar_acessos()

    treino_dados = X[:90]
    treino_marcacoes = Y[:90]

    teste_dados = X[-9:]
    teste_marcacoes = Y[-9:]

    modelo = MultinomialNB()
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    diferencas = resultado - teste_marcacoes
    acertos = [d for d in diferencas if d==0]
    total_acertos = len(acertos)
    total_elementos = len(teste_dados)

    taxa_acerto = 100.0 * total_acertos/total_elementos
    print (taxa_acerto)
