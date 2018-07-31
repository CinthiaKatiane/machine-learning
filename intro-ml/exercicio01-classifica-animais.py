from sklearn.naive_bayes import MultinomialNB

#e gordinho? tem perna curta? late?
prc1 = [1,1,0]
pcr2 = [1,1,0]
pcr3 = [1,1,0]
chr1 = [1,1,1]
chr2 = [0,1,1]
chr3 = [0,1,1]

dados = [prc1,pcr2,pcr3,chr1,chr2,chr3]
marcacoes = [1,1,1,-1,-1,-1]

teste = [[1,1,1],[1,0,0],[1,0,1]]
marcacoes_teste = [-1,1,-1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)
resultado = modelo.predict(teste)
diferencas = resultado - marcacoes_teste
acertos = [d for d in diferencas if d==0]
total_acertos = len(acertos)
total_elementos = len(marcacoes_teste)
taxa_acerto = 100 * total_acertos/total_elementos
print (taxa_acerto)
