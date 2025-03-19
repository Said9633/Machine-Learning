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


class CardioData:
    
    def __init__(self,filepath):
        
        self.df = pd.read_csv(filepath, sep=';')
        
    
    def clean_data(self):
        # omvandla ålder till år istället för dagar
        self.df['age']= (self.df['age']/365).astype(int)
        self.df.drop(columns=['id'], inplace=True)
        return self.df

        # klassafering for blood tryck 
    def blood_pressure(self):

        self.df = self.df[(self.df['ap_hi'] >= 60) & (self.df['ap_hi']<= 250)]
        self.df = self.df[(self.df['ap_lo'] >= 40) & (self.df['ap_lo']<= 200)]

        blood_cat = {
            'Healty': (self.df['ap_hi']< 120)& (self.df['ap_lo']<80), 
            'Elevated':(self.df['ap_hi'].between(120, 129))& (self.df['ap_lo']<80), 
            'Stage 1 hypertension':(self.df['ap_hi'].between(130, 139)) & (self.df['ap_lo'].between(80, 89)),
            'Stage 2 hypertension':(self.df['ap_hi']>= 140)| (self.df['ap_lo']>=90),
            'Hypertension crisis':(self.df['ap_hi']>180)| (self.df['ap_lo'] >120)
            }
        self.df['blood_category'] ='Unknown'
        for category, condition in blood_cat.items():
            self.df.loc[condition, 'blood_category']= category

        self.df.loc[self.df['blood_category'] == 'Unknown', 'blood_category'] = 'Hypertension'

        return self.df
        

        # beräkning för BMI
    def calculate_bmi(self):
        self.df['BMI']= self.df['weight']/((self.df['height']/100)**2)

        bmi_categories = {
            'Under_weight':self.df['BMI']< 18.5,
            'Normal_range':self.df['BMI'].between(18.5, 25), 
            'Over_weight':self.df['BMI'].between(25, 30), 
            'Obese (class I)':self.df['BMI'].between(30, 35),
            'Obese (class II)':self.df['BMI'].between(35, 40),
            'Obese (class III)':self.df['BMI'] >= 40

        }

        self.df['BMI_category'] = 'Unknown'
        for category, mask in bmi_categories.items():
            self.df.loc[mask, 'BMI_category']= category

        return self.df
    # Skapa två dataset 
    def data_set (self):
        df1 = self.df.copy()
        df1 = df1.drop(['ap_hi', 'ap_lo', 'height','weight', 'BMI'], axis=1)
        df1 = pd.get_dummies(df1, columns=['blood_category', 'BMI_category', 'gender'])
        
        df2 = self.df.copy()
        df2 = df2.drop(['blood_category', 'BMI_category', 'height', 'weight'], axis=1)
        df2 = pd.get_dummies(df2, columns=['gender'])
        return df1, df2
    
    def train_model(self, df):
        X = df.drop(columns=['cardio'], axis=1)
        y = df['cardio']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train =scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        print("Random Forest-Accuracy:", accuracy_score(y_test, rf_pred))
        print("\n Random Forest- Classification Report:")
        print(classification_report(y_test, rf_pred))
        


        kn = KNeighborsClassifier(n_neighbors=5)
        kn.fit(X_train, y_train)
        kn_pred = kn.predict(X_test)
        print("KNN-Accuracy:", accuracy_score(y_test, kn_pred))
        print("\n KNN- Classification Report:")
        print(classification_report(y_test, kn_pred))

        
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        print("Logistic Regression-Accuracy:", accuracy_score(y_test, lr_pred))
        print("\n Logistic Regression - Classification Report:")
        print(classification_report(y_test, lr_pred))

        


        return lr_pred, rf_pred , kn_pred, y_test
    



