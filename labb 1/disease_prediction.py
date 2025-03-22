import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier


class CardioData:
    
    def __init__(self,filepath):
        # ladda in dataset och ta bort id kolumn
        self.df = pd.read_csv(filepath, sep=';')
        self.df.drop(columns=['id'], inplace=True)

        
    def clean_data(self):
        # konvert ålder från dagar år
        self.df['age']= (self.df['age']/365).astype(int)
        return self.df
     

    def feature_engineering(self):
        # klassafering for blood tryck 
        self.df = self.df[(self.df['ap_hi'] >= 0) & (self.df['ap_hi']<= 250)]
        self.df = self.df[(self.df['ap_lo'] >= 0) & (self.df['ap_lo']<= 200)]

        blood_conditions = [
            (self.df['ap_hi']< 120)& (self.df['ap_lo']<80), 
            (self.df['ap_hi'].between(120, 129))& (self.df['ap_lo']<80), 
            (self.df['ap_hi'].between(130, 139)) & (self.df['ap_lo'].between(80, 89)),
            (self.df['ap_hi']>= 140)| (self.df['ap_lo']>=90),
            (self.df['ap_hi']>180)| (self.df['ap_lo'] >120)
        ]
        blood_categories = ['Healthy', 'Elevated', 'Stage 1 hypertension', 'Stage 2 hypertension', 'Hypertension crisis']
        self.df['blood_category'] = np.select(blood_conditions, blood_categories, default='Hypertension')

        # beräkning för BMI och klassifiera vikt
        self.df['BMI']= self.df['weight']/((self.df['height']/100)**2)

        bmi_conditions = [
            (self.df['BMI']< 18.5),
            (self.df['BMI'].between(18.5, 25)), 
            (self.df['BMI'].between(25, 30)), 
            (self.df['BMI'].between(30, 35)),
            (self.df['BMI'].between(35, 40)),
            (self.df['BMI'] >= 40)

        ]
        categories = ['Under_weight', 'Normal_range','Over_weight','Obese (class I)', 'Obese (class II)', 'Obese (class III)']
        
        self.df['BMI_category'] = np.select(bmi_conditions, categories, default='Unknown')

        return self.df
    
    def data_set (self):
        # Skapa nya dataset
        df1 = self.df.copy()
        df1 = df1.drop(['ap_hi', 'ap_lo', 'height','weight', 'BMI'], axis=1)
        df1 = pd.get_dummies(df1, columns=['blood_category', 'BMI_category', 'gender'])
        
        df2 = self.df.copy()
        df2 = df2.drop(['blood_category', 'BMI_category', 'height', 'weight'], axis=1)
        df2 = pd.get_dummies(df2, columns=['gender'])
        return df1, df2
    
    def preprocess_data(self, df):
        X = df.drop(columns=['cardio'], axis=1)
        y = df['cardio']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        sc = StandardScaler()
        X_train_sc =sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_test)
        return X_train_sc , X_test_sc, y_train, y_test

    
    def train_model(self, df):

        X_train , X_test, y_train, y_test = self.preprocess_data(df)
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42), 
            'KNN': KNeighborsClassifier(n_neighbors=5), 
            'Logistic Regression': LogisticRegression(max_iter=1000), 
            'SVM': SVC(kernel ='linear', probability=True)
        }

        for name , model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            print(f"\n {name}")
            print(f"\n Accuracy: {accuracy_score(y_test, y_pred)}")
            print(f"\n Confusion Matrix':{confusion_matrix(y_test, y_pred)}")
            print(f"\n Classification Report': {classification_report(y_test, y_pred)}")
               
    
    def tune_hyperparameters(self, model, param_grid, X_train, y_train):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_   
     
        print(f"Best parameterea for {model.__class__.__name__}: {grid_search.best_params_}")
        return best_model  
    
     
    def tune_models(self,df):
        # tunear hyperparametrar för modeller
        X_train , X_test, y_train, y_test = self.preprocess_data(df)
        Param_grids = {
            'Random Forest': {
                'model' :RandomForestClassifier(random_state=42), 
                'params' : {
                    'n_estimators' :[50, 100], 
                    'max_depth': [None, 10], 
                    'min_samples_split' : [2, 5], 
                    'min_samples_leaf': [1, 2],
                }

            }, 
            'KNN': {
                'model' : KNeighborsClassifier(), 
                'params' : {
                    'n_neighbors': [3, 5], 
                    'weights': ['uniform', 'distance']
                }
            
            }, 
            'Logistic Regression' : {
                'model' : LogisticRegression(random_state=42, max_iter=500),
                'params':{
                    'C': [0.1, 1],
                    'penalty': ['l2'],
                    'solver': ['saga','liblinear']
                }

            }, 
            'SVM': {
                'model': SVC(probability=True),
                'params':{
                    'C': [0.1, 1],
                    'kernel':['linear',]
                }
            }
        }
         
        best_models = {}
        for name, Param_grid in Param_grids.items():
            print(f" Tuning {name}")

            best_models[name] = self.tune_hyperparameters(Param_grid['model'], Param_grid['params'], X_train, y_train)
        return best_models
    
    def ensemble_model(self, df, best_models):

        X = df.drop(columns=['cardio'])
        y = df['cardio']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ensemble = VotingClassifier(
            estimators=[
                ('lr', best_models['Logistic Regression']),
                ('rf', best_models['Random Forest']),
                ('SVM', best_models['SVM']),
                ('knn', best_models['KNN'])
            ],
            voting='soft'
        )

        ensemble.fit(X_scaled, y)
        y_pred = ensemble.predict(X_scaled)
        print("Ensemble_modell: ")
        print(f"Accuracy:{accuracy_score(y,y_pred):.4f}")
        print("Classification Report:\n", classification_report(y, y_pred))
        return ensemble
    