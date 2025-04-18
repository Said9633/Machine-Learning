

## Syfte: 
   Syftet med denna laboration är att bygga ett filmrekommendationssystem som föreslår fem liknande filmer till användaren, baserat på en inmatad filmtitel.

## Metod:

   Labrationen började med att ladda ner datasetet och läsa in filerna med hjälp av pandas. En metod (__init__) skapades för att läsa in filerna. En funktion (clean_data) användes för att ta bort onödliga tecken från titlar och taggar. Funktionen (clean_title) användes för att kombinera titel, genre och taggar till en textbaserad representation av varje film. 
   En (fit) funktion användes för att träna systemet, den räknar ut genomsnittligt betyg för varje film och filtrerar bort filmer med låga betyg(< 3). 'TfidfVectorizer' användes för att omvandla texten till numeriska vektorer som representerar varje film. (KNN) med cosinuslikhet för att mäta hur lika filmer är.
   Funktionen (search) tar emot input från användaren och returnerar de fem mest liknande filmer. Funktionen (display) som visar resultaten. 

## Diskussion:
   Systemet gav ganska bra resultat. Det skulle kunna förbättras ytterligare genom att testa fler modeller eller förbättra visualiseringar, exempelvis genom att bygga ett interaktivt dashboard. 