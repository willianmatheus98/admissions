from time import time
import pandas as pd
data_frames = pd.read_csv('Admission_Predict_Ver1.1.csv')

#print(data_frames)

X_df = data_frames[['GRE_Score','TOEFL_Score','University_Rating','SOP' ,'LOR', 'CGPA','Research']] #colunas do dataset
Y_df = data_frames.Chance #vai ser a coluna que vai classificar os dados

Y = list()

for i, value in enumerate(Y_df):
    if ( Y_df[i] >= 0.8): #classificados
        Y.append(2)
    elif (Y_df[i] >= 0.5): #classificáveis
        Y.append(1)
    else:
        Y.append(0) #não classificados

Xdummies_df = pd.get_dummies(X_df)

X =  Xdummies_df.values         #Devolve os Dummies em Arrays

#codigo para efetuar o treino e teste
from sklearn.model_selection import train_test_split
treino_dados, teste_dados, treino_marcacoes, teste_marcacoes = train_test_split(X, Y, test_size=0.2) #os dados de treino são 20% do dataset

#print( treino_dados )
#print( treino_marcacoes )



from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import  MultinomialNB

clf_dict = {'log reg': LogisticRegression(solver='liblinear'),
            'naive bayes': GaussianNB(),
            'random forest': RandomForestClassifier(n_estimators=100),
            'knn': KNeighborsClassifier(),
            'linear svc': LinearSVC(),
            'ada boost': AdaBoostClassifier(n_estimators=100),
            'gradient boosting': GradientBoostingClassifier(n_estimators=100),
            'CART': DecisionTreeClassifier(),
            'Multinomial Naive Bayes': MultinomialNB()}

for name, clf in clf_dict.items():
    t0 = time()
    model = clf.fit(treino_dados, treino_marcacoes)
    pred = model.predict(teste_dados)
    model.score(teste_dados, teste_marcacoes)
    print('Precisão {}:'.format(name), model.score(teste_dados, teste_marcacoes))
    print("Tempo gasto:", round(time() - t0, 3), "s")

print("Fim da execução!")