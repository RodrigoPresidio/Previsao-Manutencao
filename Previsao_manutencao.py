# Import
import pickle
import sklearn as sk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Carregando os dados
df = pd.read_csv("dataset.csv")

# Shape
df.shape

# Resumo estatístico
df.describe()

# Verificando valores ausentes
df.isna().sum().sum()

# Esta função calcula a prevalência da classe positiva (label = 1)
def calcula_prevalencia(y_actual):
    return sum(y_actual) / len(y_actual)
print("Prevalência da classe positiva: %.3f"% calcula_prevalencia(df["VARIAVEL_ALVO"].values))

df.value_counts('VARIAVEL_ALVO')

# Preparando o dataset somente com os dados de interesse
collist = df.columns.tolist()
cols_input = collist[0:178]
df_data = df[cols_input + ["VARIAVEL_ALVO"]]

# Checando se temos colunas duplicadas nos dados de entrada
dup_cols = set([x for x in cols_input if cols_input.count(x) > 1])
print(dup_cols)
assert len(dup_cols) == 0, "há colunas duplicadas"

# Checando se temos colunas duplicadas no dataset final
cols_df_data = list(df_data.columns)
dup_cols = set([x for x in cols_df_data if cols_df_data.count(x) > 1])
print(dup_cols)
assert len(dup_cols) == 0, "há colunas duplicadas"
df_data.head()

# Gerando amostras aleatórias dos dados
df_data = df_data.sample(n = len(df_data))
len(df_data)

# Ajustando os índices do dataset
df_data = df_data.reset_index(drop = True)

# Gera um índice para a divisão
df_valid_teste = df_data.sample(frac = 0.3)
print("Tamanho da divisão de validação / teste: %.1f" % (len(df_valid_teste) / len(df_data)))

# Fazendo a divisão 70/15/15
# Dados de teste
df_teste = df_valid_teste.sample(frac = 0.5)

# Dados se validação
df_valid = df_valid_teste.drop(df_teste.index)

# Dados de treino
df_treino = df_data.drop(df_valid_teste.index)

# Verifique a prevalência de cada subconjunto
print("Teste(n = %d): %.3f" % (len(df_teste), calcula_prevalencia(df_teste.VARIAVEL_ALVO.values)))
print("Validação(n = %d): %.3f" % (len(df_valid), calcula_prevalencia(df_valid.VARIAVEL_ALVO.values)))
print("Treino(n = %d): %.3f" % (len(df_treino), calcula_prevalencia(df_treino.VARIAVEL_ALVO.values)))
print('Todas as amostras (n = %d)'%len(df_data))
assert len(df_data) == (len(df_teste) + len(df_valid) + len(df_treino)), 'algo saiu errado'

# Cria um índice
rows_pos = df_treino.VARIAVEL_ALVO == 1

# Define valores positivos e negativos do índice
df_train_pos = df_treino.loc[rows_pos]
df_train_neg = df_treino.loc[~rows_pos]

# Valor mínimo
n = np.min([len(df_train_pos), len(df_train_neg)])

# Obtém valores aleatórios para o dataset de treino
df_treino_final = pd.concat([df_train_pos.sample(n = n, random_state = 64), 
                             df_train_neg.sample(n = n, random_state = 64)], 
                            axis = 0, 
                            ignore_index = True)

# Amostragem
df_treino_final = df_treino_final.sample(n = len(df_treino_final), random_state = 64).reset_index(drop = True)

print('Balanceamento em Treino(n = %d): %.3f'%(len(df_treino_final), 
                                               calcula_prevalencia(df_treino_final.VARIAVEL_ALVO.values)))

# Salvar todos os datasets em disco no formato csv.
df_treino.to_csv('dados_treino.csv', index = False)
df_treino_final.to_csv('dados_treino_final.csv', index = False)
df_valid.to_csv('dados_valid.csv', index = False)
df_teste.to_csv('dados_teste.csv', index = False)

# Salvar os dados de entrada (colunas preditoras) para facilitar a utilização mais tarde
pickle.dump(cols_input, open('cols_input.sav', 'wb'))

# X
X_treino = df_treino_final[cols_input].values
X_valid = df_valid[cols_input].values

# Y
y_treino = df_treino_final['VARIAVEL_ALVO'].values
y_valid = df_valid['VARIAVEL_ALVO'].values

# Print
print('Shape dos dados de treino:', X_treino.shape, y_treino.shape)
print('Shape dos dados de validação:', X_valid.shape, y_valid.shape)

# Cria o objeto de Padronização
scaler = StandardScaler()
scaler.fit(X_treino)

# Salva o objeto em disco e carrega para usamos adiante
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))
scaler = pickle.load(open(scalerfile, 'rb'))

# Aplica a padronização nas matrizes de dados
X_treino_tf = scaler.transform(X_treino)
X_valid_tf = scaler.transform(X_valid)

# Função para calcular a especificidade
def calc_especificidade(y_actual, y_pred, thresh):
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual ==0)

# Função para gerar relatório de métricas
def relatorio(y_actual, y_pred):
    
    auc = roc_auc_score(y_actual, y_pred)
    
    accuracy = accuracy_score(y_actual, (y_pred > 0.5))
    
    recall = recall_score(y_actual, (y_pred > 0.5))
    
    precision = precision_score(y_actual, (y_pred > 0.5))
    
    especificidade = calc_especificidade(y_actual, y_pred, 0.5)
    
    print('AUC:%.3f'%auc)
    print('Acurácia:%.3f'%accuracy)
    print('Recall:%.3f'%recall)
    print('Precisão:%.3f'%precision)
    print('Especificidade:%.3f'%especificidade)
    print(' ')
    
    return auc, accuracy, recall, precision, especificidade

# Construção do modelo 1

# Cria o classificador (objeto)
lr = LogisticRegression(max_iter = 500, random_state = 142)

# Treina e cria o modelo
modelo_v1 = lr.fit(X_treino_tf, y_treino)

# Previsões 
y_train_preds_v1 = modelo_v1.predict_proba(X_treino_tf)[:,1]
y_valid_preds_v1 = modelo_v1.predict_proba(X_valid_tf)[:,1]

print('\nRegressão Logística\n')

print('Treinamento:\n')
v1_train_auc, v1_train_acc, v1_train_rec, v1_train_prec, v1_train_spec = relatorio(y_treino, y_train_preds_v1)

print('Validação:\n')
v1_valid_auc, v1_valid_acc, v1_valid_rec, v1_valid_prec, v1_valid_spec = relatorio(y_valid, y_valid_preds_v1)

# Construção do modelo 2

# Cria o classificador (objeto)
nb = GaussianNB()

# Treina e cria o modelo
modelo_v2 = nb.fit(X_treino_tf, y_treino)

# Previsões
y_train_preds_v2 = modelo_v2.predict_proba(X_treino_tf)[:,1]
y_valid_preds_v2 = modelo_v2.predict_proba(X_valid_tf)[:,1]

print('\nNaive Bayes\n')

print('Treinamento:\n')
v2_train_auc, v2_train_acc, v2_train_rec, v2_train_prec, v2_train_spec = relatorio(y_treino, y_train_preds_v2)

print('Validação:\n')
v2_valid_auc, v2_valid_acc, v2_valid_rec, v2_valid_prec, v2_valid_spec = relatorio(y_valid, y_valid_preds_v2)

# Construção do modelo 3

# Cria o classificador (objeto)
rf = RandomForestClassifier()

# Treina e cria o modelo
modelo_v3 = rf.fit(X_treino_tf, y_treino)

# Previsões
y_train_preds_v3 = modelo_v3.predict_proba(X_treino_tf)[:,1]
y_valid_preds_v3 = modelo_v3.predict_proba(X_valid_tf)[:,1]

print('\nRandom Forest\n')

print('Treinamento:\n')
v3_train_auc, v3_train_acc, v3_train_rec, v3_train_prec, v3_train_spec = relatorio(y_treino, y_train_preds_v3)

print('Validação:\n')
v3_valid_auc, v3_valid_acc, v3_valid_rec, v3_valid_prec, v3_valid_spec = relatorio(y_valid, y_valid_preds_v3)

# Construção do modelo 4

# Cria o classificador (objeto)
xgbc = XGBClassifier()

# Treina e cria o modelo
modelo_v4 = xgbc.fit(X_treino_tf, y_treino)

# Previsões
y_train_preds_v4 = modelo_v4.predict_proba(X_treino_tf)[:,1]
y_valid_preds_v4 = modelo_v4.predict_proba(X_valid_tf)[:,1]

print('\nXGB Classifier\n')

print('Treinamento:\n')
v4_train_auc, v4_train_acc, v4_train_rec, v4_train_prec, v4_train_spec = relatorio(y_treino, y_train_preds_v4)

print('Validação:\n')
v4_valid_auc, v4_valid_acc, v4_valid_rec, v4_valid_prec, v4_valid_spec = relatorio(y_valid, y_valid_preds_v4)

# Cria o classificador
xgbc = XGBClassifier()

# Configura a validação cruzada
# Por exemplo, usando 5 divisões e a métrica de área sob a curva ROC (AUC)
n_splits = 5
score = 'roc_auc'

# Realiza a validação cruzada
cv_scores = cross_val_score(xgbc, X_treino_tf, y_treino, cv = n_splits, scoring = score)

# Exibe os resultados
print(f"Validação Cruzada com {n_splits} divisões")
print(f"Score AUC em Cada Divisão: {cv_scores}")
print(f"Média de Score AUC: {np.mean(cv_scores)}")

# Cria o classificador
rf = RandomForestClassifier()

# Configura a validação cruzada
# Por exemplo, usando 5 divisões e a métrica de área sob a curva ROC (AUC)
n_splits = 5
score = 'roc_auc'

# Realiza a validação cruzada
cv_scores = cross_val_score(rf, X_treino_tf, y_treino, cv = n_splits, scoring = score)

# Exibe os resultados
print(f"Validação Cruzada com {n_splits} divisões")
print(f"Score AUC em Cada Divisão: {cv_scores}")
print(f"Média de Score AUC: {np.mean(cv_scores)}")

# Define o classificador
xgbc = XGBClassifier()

# Define o espaço de hiperparâmetros para a otimização
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9]
}

# Configura o GridSearchCV
grid_search = GridSearchCV(xgbc, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1)

# Realiza a busca pelos melhores hiperparâmetros
grid_search.fit(X_treino_tf, y_treino)

# Melhores hiperparâmetros encontrados
best_params = grid_search.best_params_

# Treina o modelo com os melhores hiperparâmetros
modelo_v5 = grid_search.best_estimator_

# Previsões com o modelo otimizado
y_train_preds_optimized_v5 = modelo_v5.predict_proba(X_treino_tf)[:,1]
y_valid_preds_optimized_v5 = modelo_v5.predict_proba(X_valid_tf)[:,1]

# Avaliação do modelo otimizado
print('\nXtreme Gradient Boosting Classifier - Otimizado\n')
print('Melhores hiperparâmetros:', best_params)

print('\nTreinamento:\n')
v5_train_auc, v5_train_acc, v5_train_rec, v5_train_prec, v5_train_spec = relatorio(y_treino, y_train_preds_optimized_v5)

print('Validação:\n')
v5_valid_auc, v5_valid_acc, v5_valid_rec, v5_valid_prec, v5_valid_spec = relatorio(y_valid, y_valid_preds_optimized_v5)

# Define o classificador
rf = RandomForestClassifier()

# Define o espaço de hiperparâmetros para a otimização
param_grid = {
    'max_depth': [5, 6, 7],
    'n_estimators': [100],
}

# Configura o GridSearchCV
grid_search = GridSearchCV(rf, param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1)

# Realiza a busca pelos melhores hiperparâmetros
grid_search.fit(X_treino_tf, y_treino)

# Melhores hiperparâmetros encontrados
best_params = grid_search.best_params_

# Treina o modelo com os melhores hiperparâmetros
modelo_v6 = grid_search.best_estimator_

# Previsões com o modelo otimizado
y_train_preds_optimized_v6 = modelo_v6.predict_proba(X_treino_tf)[:,1]
y_valid_preds_optimized_v6 = modelo_v6.predict_proba(X_valid_tf)[:,1]

# Avaliação do modelo otimizado
print('\nRandom Forest - Otimizado\n')
print('Melhores hiperparâmetros:', best_params)

print('\nTreinamento:\n')
v6_train_auc, v6_train_acc, v6_train_rec, v6_train_prec, v6_train_spec = relatorio(y_treino, y_train_preds_optimized_v6)

print('Validação:\n')
v6_valid_auc, v6_valid_acc, v6_valid_rec, v6_valid_prec, v6_valid_spec = relatorio(y_valid, y_valid_preds_optimized_v6)

# Cria um dataframe com as métricas calculadas
df_results = pd.DataFrame({'classificador':['RL','RL','NB','NB','XGB','XGB','XGB_O','XGB_O','RF','RF','RF_O','RF_O'],
                           'data_set':['treino','validação'] * 6,
                           'auc':[v1_train_auc,
                                  v1_valid_auc,
                                  v2_train_auc,
                                  v2_valid_auc,
                                  v3_train_auc,
                                  v3_valid_auc,
                                  v4_train_auc,
                                  v4_valid_auc,
                                  v5_train_auc,
                                  v5_valid_auc,
                                  v6_train_auc,
                                  v6_valid_auc],
                           'accuracy':[v1_train_acc,
                                       v1_valid_acc,
                                       v2_train_acc,
                                       v2_valid_acc,
                                       v3_train_acc,
                                       v3_valid_acc,
                                       v4_train_acc,
                                       v4_valid_acc,
                                       v5_train_acc,
                                       v5_valid_acc,
                                       v6_train_acc,
                                       v6_valid_acc],
                           'recall':[v1_train_rec,
                                     v1_valid_rec,
                                     v2_train_rec,
                                     v2_valid_rec,
                                     v3_train_rec,
                                     v3_valid_rec,
                                     v4_train_rec,
                                     v4_valid_rec,
                                     v5_train_rec,
                                     v5_valid_rec,
                                     v6_train_rec,
                                     v6_valid_rec],
                           'precision':[v1_train_prec,
                                        v1_valid_prec,
                                        v2_train_prec,
                                        v2_valid_prec,
                                        v3_train_prec,
                                        v3_valid_prec,
                                        v4_train_prec,
                                        v4_valid_prec,
                                        v5_train_prec,
                                        v5_valid_prec,
                                        v6_train_prec,
                                        v6_valid_prec],
                           'specificity':[v1_train_spec,
                                          v1_valid_spec,
                                          v2_train_spec,
                                          v2_valid_spec,
                                          v3_train_spec,
                                          v3_valid_spec,
                                          v4_train_spec,
                                          v4_valid_spec,
                                          v5_train_spec,
                                          v5_valid_spec,
                                          v6_train_spec,
                                          v6_valid_spec]})

# Construção do Plot
sns.set_style("whitegrid")
plt.figure(figsize = (16, 8))

# Gráfico de barras
ax = sns.barplot(x = 'classificador', y = 'auc', hue = 'data_set', data = df_results)
ax.set_xlabel('Classificador', fontsize = 15)
ax.set_ylabel('AUC', fontsize = 15)
ax.tick_params(labelsize = 15)

# Legenda
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., fontsize = 15)
plt.show();

# Tabela de comparação dos modelos
df_results

# Tabela de comparação dos modelos somente com métricas em validação e ordenado por AUC
df_results[df_results['data_set'] == 'validação'].sort_values(by = 'auc', ascending = False)

# Grava o modelo em disco
pickle.dump(modelo_dsa_v4, open('melhor_modelo_dsa.pkl', 'wb'), protocol = 4)

# Carrega o modelo, as colunas e o scaler
melhor_modelo = pickle.load(open('melhor_modelo_dsa.pkl','rb'))
cols_input = pickle.load(open('cols_input.sav','rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Carrega os dados
df_train = pd.read_csv('dados_treino.csv')
df_valid= pd.read_csv('dados_valid.csv')
df_test= pd.read_csv('dados_teste.csv')

# Cria matrizes x e y

# X
X_train = df_train[cols_input].values
X_valid = df_valid[cols_input].values
X_test = df_test[cols_input].values

# Y
y_train = df_train['VARIAVEL_ALVO'].values
y_valid = df_valid['VARIAVEL_ALVO'].values
y_test = df_test['VARIAVEL_ALVO'].values

# Aplica a transformação nos dados
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
X_test_tf = scaler.transform(X_test)

# Calcula as probabilidades
y_train_preds = melhor_modelo.predict_proba(X_train_tf)[:,1]
y_valid_preds = melhor_modelo.predict_proba(X_valid_tf)[:,1]
y_test_preds = melhor_modelo.predict_proba(X_test_tf)[:,1]

print('\nTreinamento:\n')
train_auc, train_accuracy, train_recall, train_precision, train_specificity = relatorio(y_train, y_train_preds)

print('\nValidação:\n')
valid_auc, valid_accuracy, valid_recall, valid_precision, valid_specificity = relatorio(y_valid, y_valid_preds)

print('\nTeste:\n')
test_auc, test_accuracy, test_recall, test_precision, test_specificity = relatorio(y_test, y_test_preds)

# Calcula a curva ROC nos dados de treino
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)

# Calcula a curva ROC nos dados de validação
fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

# Calcula a curva ROC nos dados de teste
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

# Plot
plt.figure(figsize=(16,10))
plt.plot(fpr_train, tpr_train, 'r-', label = 'AUC em Treino: %.3f' % auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-', label = 'AUC em Validação: %.3f' % auc_valid)
plt.plot(fpr_test, tpr_test, 'g-', label = 'AUC em Teste: %.3f' % auc_test)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.legend()
plt.show()

# Carregando novos dados
nova_maquina = pd.read_csv('novos_dados.csv')

# Aplicamos a padronização aos novos dados de entrada
nova_maquina_scaled = scaler.transform(nova_maquina)

# Previsão de classe
previsao = melhor_modelo.predict(nova_maquina_scaled)

if previsao == 0:
    print("O equipamento não precisa de manutenção")
else:
    print("O equipamento precisa de manutenção")