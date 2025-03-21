. Syfte:
    Syftet med denna labrationen är att analysera ett dataset som innehåller information och medicinsk data för 700000 individer. Maskininlärningsmodeller användes för att identifera vilka faktorer som har störst påverkan på risken för hjärt-kärlsjukdom.


. Data-analys(EDA):
    Seaborn och Matplotlib användes för att analysera datasetet. Sjukdomsfördelningen visade sig vara jämnt fördelad mellan individer med och utan hjärt-kärlsjukdom. Analysen av kolesterolnivåer visade att den största andelen individer, cirka 75 %, hade normala kolesterolnivåer, medan 13,5 % hade låg kolesterolnivå. Vid analys av sambandet mellan ålder och sjukdom framkom att äldre individer har en högre risk för hjärt-kärlsjukdom.
    Rökning och alkohol visade sig inte vara den största faktoren bakom risken för sjukdomen, BMI och vikt var de största faktorerna för hjärt-kärlsjukdom medan för de individer som har normal range för BMI vad de flesta friska, även för blodtryck för individer med "Stage 2 Hypertension" hade högre sjukdomsförekomst. Analysen visade också att majoriteten av fysiskt aktiva personer var friska, medan de som inte var fysiskt aktiva hade en högre riskt för sjukdomen. 


. Modeller:
    Fyra modeller användes i denna laberation KNN, Random Forest, Logistisk Regression och Support Vector Machine (SVM). Modellerna tränades på två dataset med olika feature-engeingering. Den modellen som presnterade de bästa resultaten var logistisk regression med en noggranhet på båda datan 72-73 %. 
    GridSearchCV användes för att hitta bästa hyperparametrer.
    C = 0.1
    Penalty = l2
    solver = saga 


. Resultat:
    Logistis Regression valdes som bästa modellen, resultaten var:
    Accuracy : 72,3 %
    F1-score för friska: 0.75
    F1-score för sjucka: 0.70


. Diskussion:
    Modellen hade inte en extremt hög noggrannhet, vilket kan bero på de feature som valdes. Till exempel visade sig alkohol och rökning inte vara starka faktorer, medan vikt och BMI hade en stark koppling till sjukdomen. En annan möjlig förbättring är att testa fler modeller och justera hyperparametrarna för att öka noggrannhet.