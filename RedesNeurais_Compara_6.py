#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Case: Cobrança - Comparação de Técnicas
## Autor: Prof. Roberto Angelo
## Objetivo: Ajuste de Redes Neurais Artificiais
##------------------------------------------------------------------------

# Bibliotecas padrão
import numpy as np
import pandas as pd

## Carregando os dados
dataset = pd.read_csv('Case_cobranca.csv') 

#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis
#------------------------------------------------------------------------------------------
## Tratamento de nulos no alvo --- Tempo de Atraso - transformação para alvo binário (>90 dias) 
dataset['ALVO']   = [0 if np.isnan(x) or x > 90 else 1 for x in dataset['TEMP_RECUPERACAO']]
## Tratamento de nulos e normalização --- Variáveis de entrada numéricas
dataset['PRE_IDADE']        = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] # Trata mínimo
dataset['PRE_IDADE']        = [1. if x > 76 else (x-18)/(76-18) for x in dataset['PRE_IDADE']] # Trata máximo por percentil 99 e coloca na fórmula
dataset['PRE_QTDE_DIVIDAS'] = [0.  if np.isnan(x) else x/16. for x in dataset['QTD_DIVIDAS']] # retirada de outlier com percentil 99 e normalização     
##--- Dummies - transformação de atributos categóricos em numéricos e tratamanto de nulos ---------------
dataset['PRE_NOVO']         = [1 if x=='NOVO'                      else 0 for x in dataset['TIPO_CLIENTE']]    
dataset['PRE_TOMADOR_VAZIO']= [1 if x=='TOMADOR' or str(x)=='nan'  else 0 for x in dataset['TIPO_CLIENTE']]                        
dataset['PRE_CDC']          = [1 if x=='CDC'                       else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_PESSOAL']      = [1 if x=='PESSOAL'                   else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_SEXO_M']       = [1 if x=='M'                         else 0 for x in dataset['CD_SEXO']]


##------------------------------------------------------------
## Separando em dados de treinamento e teste
##------------------------------------------------------------
y = dataset['ALVO']              # Carrega alvo ou dataset.iloc[:,7].values
X = dataset.iloc[:, 8:15].values # Carrega colunas 8, 9, 10, 11, 12, 13 e 14 (a 15 não existe até este momento)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 175)

#---------------------------------------------------------------------------
## Ajustando a RNA - Aprendizado supervisionado  
#---------------------------------------------------------------------------
#Rede Neural com dados de treinamento 
from sklearn.neural_network import MLPClassifier 
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(2, 7), learning_rate='constant',
       learning_rate_init=0.1, max_iter=1000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=175, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.2, verbose=False,
       warm_start=False)
RNA.fit(X_train, y_train)

#---------------------------------------------------------------------------
## Previsão usando todos os conjuntos (treinamento e teste)
#---------------------------------------------------------------------------
# Rede Neural
y_pred_train_RNA_P = RNA.predict_proba(X_train)[:,1]
y_pred_test_RNA_P  = RNA.predict_proba(X_test)[:,1]

##-----------------------------------------------------------------
## Cálculo dos erros da classificação e Matriz de confusão da RNA
##-----------------------------------------------------------------
Erro_RNA_MSE_TRAIN = np.mean((y_pred_train_RNA_P - y_train) ** 2)
Erro_RNA_MSE_TEST  = np.mean((y_pred_test_RNA_P - y_test) ** 2)

print("__________________________________________")
print()
print('Rede Neural TRN - MSE:', Erro_RNA_MSE_TRAIN)
print('Rede Neural TST - MSE:', Erro_RNA_MSE_TEST)
print()
print("__________________________________________")