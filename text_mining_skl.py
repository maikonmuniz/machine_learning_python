# Bibliotecas padrão  e carga de dados
import pandas as pd
dataset = pd.read_csv('Amazon_Reviews_10000.txt', sep=';')

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+') # Expressão regular para remover simbolos
cv = CountVectorizer(   analyzer='word',lowercase=True, stop_words='english',min_df=1,
                        ngram_range = (1,1), tokenizer = token.tokenize)
text_counts = cv.fit_transform(dataset['Text'])
# print(cv.vocabulary_)

##------------------------------------------------------------
## Separa os dados em treinamento e teste
##------------------------------------------------------------
y = dataset['Sentiment']   # Carrega alvo
X = text_counts            # Carrega as colunas geradas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=474)

#---------------------------------------------------------------------------
## Ajusta modelo Naive Bayes com treinamento - Aprendizado supervisionado  
#---------------------------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB
NaiveB = MultinomialNB()
NaiveB.fit(X_train, y_train)

#---------------------------------------------------------------------------
## Previsão usando os dados de teste
#---------------------------------------------------------------------------
# Naive Bayes
y_pred_test_NaiveB= NaiveB.predict(X_test)

#---------------------------------------------------------------------------
## Cálcula da Acurácia do Naive Bayes
#---------------------------------------------------------------------------
from sklearn import metrics

# Regressão logística com dados de treinamento
from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression()
LogisticReg.fit(X_train, y_train)

# Árvore de decisão com dados de treinamento
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

#Rede Neural com dados de treinamento
from sklearn.neural_network import MLPClassifier
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(3), learning_rate='constant',
       learning_rate_init=0.1, max_iter=10000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, verbose=False,validation_fraction = 0.2,
       warm_start=False)
RNA.fit(X_train, y_train)

y_pred_test_RNA  = RNA.predict(X_test)
y_pred_test_DT  = dtree.predict(X_test)
y_pred_test_RLog  = LogisticReg.predict(X_test)
# Acurácia
from sklearn.metrics import accuracy_score # importando a biblioteca para calcular a acurácia
Acuracia_RNA_Classificacao = metrics.accuracy_score(y_test, y_pred_test_RNA)
Acuracia_DT_Classificacao = metrics.accuracy_score(y_test, y_pred_test_DT)
Acuracia_RLog_Classificacao = metrics.accuracy_score(y_test, y_pred_test_RLog)

print()

print('Acurácia NaiveBayes:',metrics.accuracy_score(y_test, y_pred_test_NaiveB))


print()



print()

print('Acurácia DT:',Acuracia_DT_Classificacao)


print()

print('Acurácia RNA:',Acuracia_RNA_Classificacao)

print()

print('Acurácia Regressão Logistica:',Acuracia_RLog_Classificacao)