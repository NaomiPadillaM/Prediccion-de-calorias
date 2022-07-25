#Análisis nutrimental - predicción de calorías
#Naomi Padilla Mora
#Junio 2021

import pandas as pd
import matplotlib.pyplot as plt
#Regresión lineal sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import cross_val_score

df = pd.read_excel('nutrimental.xlsx')

df.columns = ['dia', 'tipo', 'comida', 'cal', 'grasa', 'prote', 'carb', 'sodio']
pd.set_option('display.max_columns',None)

selection = ['cal', 'grasa', 'prote', 'carb', 'sodio']
select = df[selection]
print(select,'\n')

#Variables X y Y para la regresión lineal
features = ['grasa','prote','carb','sodio']
X = df[features]

y = df['cal']

#Parte 1: train data
# separamos nuestra data dos partes para entrenar y para hacer el test.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=10)

print('********************Regresion lineal Sklearn********************\n')
model = LinearRegression()  # inciamos el modelo de LinearRegression

#Cross validation
print('Cross validation\n')
scores = cross_val_score(model, X, y,cv=5)

print("scores:\n", scores)
print('\nmean score',scores.mean())
print('standard deviation',scores.std())

model.fit(X_train,y_train)  # entrenamos el modelo con lo obtenido anteriormente

#Parte 2: test data - validación
predictions = model.predict(X_test)

#mostramos el puntaje del entrenamiento
print('\nPuntaje entrenamiento: {}\n'.format(model.score(X_train,y_train)))
print('Puntaje Test: {}\n'.format(model.score(X_test,y_test)))
print('Exactitud modelo: {}\n'.format(r2_score(y_test,predictions)))
print(model.intercept_, model.coef_) 
print('MAE',mean_absolute_error(y_test, predictions))
print('MSE',mean_squared_error(y_test, predictions))


#Una vez optimizado el modelo podemos usar toda la base de datos
model.fit(X,y)
print('MAE',mean_absolute_error(y, model.predict(X)))

#Gráfica Regresión lineal
def plot(X,y,model):
    y_pred = model.predict(X)
    data = pd.DataFrame({'cal actual':y,'cal predecidas':y_pred})
    plt.figure(figsize=(12,8))
    plt.scatter(data.index,data['cal actual'].values,label='cal actual')
    plt.scatter(data.index,data['cal predecidas'].values,label='cal predecidas')
    plt.title('Regresion lineal Sklearn',fontsize=16)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(loc='best')
    plt.show()
plot(X,y,model)

#Decision Tree Regressor
print('\n********************Decision Tree Regressor********************')
train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=.3, random_state = 0)

model = DecisionTreeRegressor(max_depth=10,random_state=1) # inciamos el modelo de Decision Tree

# Fit the model
model.fit(train_X,train_y)
predictions = model.predict(test_X)
print('MAE',mean_absolute_error(test_y, predictions))
print('test score',model.score(test_X,test_y))
print('train score',model.score(train_X,train_y))

#Gráfica decision tree
def plot(X,y,model):
    y_pred = model.predict(X)
    data = pd.DataFrame({'cal actual':y,'cal predecidas':y_pred})
    plt.figure(figsize=(12,8))
    plt.scatter(data.index,data['cal actual'].values,label='cal actual')
    plt.scatter(data.index,data['cal predecidas'].values,label='cal predecidas')
    plt.title('Decision Tree',fontsize=16)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(loc='best')
    plt.show()
plot(test_X,test_y,model)

#Optimización del modelo
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 25, 100, 200,500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y)
    print("\nMax leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

def get_mae2(max_depth, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(test_X)
    mae = mean_absolute_error(test_y, preds_val)
    return(mae)

for max_depth in [3, 5, 10, 15]:
    my_mae = get_mae2(max_depth, train_X, test_X, train_y, test_y)
    print("Max depth: %d  \t\t Mean Absolute Error:  %d" %(max_depth, my_mae)) 

model = DecisionTreeRegressor(max_depth=15,max_leaf_nodes=50,random_state=1)

# Fit the model
print('\nDecision Tree Optimizado')

model.fit(train_X,train_y)
predictions = model.predict(test_X)
print('\nMAE',mean_absolute_error(test_y, predictions))
print('R2 test',model.score(test_X,test_y))
print('R2_train',model.score(train_X,train_y))

#Gráfica decision tree optimizado
def plot(X,y,model):
    y_pred = model.predict(X)
    data = pd.DataFrame({'cal actual':y,'cal predecidas':y_pred})
    plt.figure(figsize=(12,8))
    plt.scatter(data.index,data['cal actual'].values,label='cal actual')
    plt.scatter(data.index,data['cal predecidas'].values,label='cal predecidas')
    plt.title('Decision Tree Optimizado',fontsize=16)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(loc='best')
    plt.show()
plot(test_X,test_y,model)
