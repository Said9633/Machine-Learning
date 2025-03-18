import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


class CardioData:
    
    def load_data(self,filepath):
        
        self.df = pd.read_csv(filepath, sep=';')
        return self.df
    
    def clean_data(self):
        # omvandla ålder till år istället för dagar
        self.df['age']= (self.df['age']/365).astype(int)

        # klassafering for blood tryck 
    def blood_pressure(self):
        blood_cat = {
            'Healty': (self.df['ap_hi']< 120)& (self.df['ap_lo']<80), 
            'Elevated':(self.df['ap_hi'].between(120, 129))& (self.df['ap_lo']<80), 
            'Stage 1 hypertension':(self.df['ap_hi'].between(120, 129)) & (self.df['ap_lo'].between(80, 89)),
            'Stage 2 hypertension':(self.df['ap_hi']>= 140)& (self.df['ap_lo']>=90),
            'Hypertension crisis':(self.df['ap_hi']>180)& (self.df['ap_lo'] >120)
            }
        self.df['blood_category'] ='Unkown'
        for category, condition in blood_cat.items():
            self.df.loc[condition, 'blood_category']= category

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

