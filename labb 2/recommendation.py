import pandas as pd 
import re 
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import display
import ipywidgets as widgets
from sklearn.neighbors import NearestNeighbors
class MovieRecommender:

    def __init__(self, movies_path , ratings_path , tags_path ):

        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.tags = pd.read_csv(tags_path)

    # rensar titlar och taggar
    def clean_data(self,title):
        return re.sub("[^a-zA-Z0-9 ]", "", title.lower())
    
    def clean_titles(self):
        self.movies["clean_title"] = self.movies ["title"].apply(self.clean_data)
        self.movies["clean_genres"] = self.movies["genres"].fillna("").apply(lambda x: x.replace("|", " ").lower())

        tag_df = self.tags.groupby("movieId")["tag"].apply(lambda x:" ".join(x.astype(str))).reset_index()
        tag_df["tag"]= tag_df["tag"].apply(self.clean_data)
        self.movies =  self.movies.merge(tag_df, on='movieId', how='left')
        self.movies["tag"]= self.movies["tag"].fillna("")

        self.movies["combined"] = (
            (self.movies["clean_title"] + " " )+
            (self.movies["clean_genres"] + " ")* 2+
            (self.movies["tag"]+ " ")*2
        )

    def fit(self):
        self.clean_titles()

        # beräknar genomsnittligt betyg 
        avg_rating = self.ratings.groupby("movieId")["rating"].mean()
        self.movies = self.movies.merge(avg_rating, on="movieId", how="left")
        self.movies =self.movies[self.movies["rating"]> 3].copy()

        # bygger TF_IDF vektorer 
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.tfidf = self.vectorizer.fit_transform(self.movies["combined"])

        self.movies["scaled_rating"] = (self.movies["rating"] - self.movies["rating"].min()) /(
            self.movies["rating"].max() - self.movies["rating"].min()
        )

        self.knn = NearestNeighbors(metric='cosine', algorithm='brute')
        self.knn.fit(self.tfidf)
    

    def search(self, query, top_n= 5):

        query = self.clean_data(query)
        query_vec = self.vectorizer.transform([query])
        distance , indices = self.knn.kneighbors(query_vec, n_neighbors=top_n*2)

        results= self.movies.iloc[indices[0]].copy()
        results["similarity"] = 1- distance[0]

        # justera likhetspöng baserat på titlar 
        results.loc[results["clean_title"].str.startswith(query), "similarity"] += 1
        results.loc[results["clean_title"].str.contains(query), "similarity"] += 0.5

        results["similarity"] += results["scaled_rating"] *0.2
        # returnerar topp N resultat
        return results.sort_values(by="similarity", ascending=False).head(top_n)[
            ["title", "genres", "rating", "similarity"]
        ].reset_index(drop=True)
    

    def display(self):
        movies_input = widgets.Text(
            placeholder ='Write a movietitle',
            description = 'Movies:', 
            layout = widgets.Layout(width='50%')
        )
       
        output_area = widgets.Output()
        
        def on_type(data):
            with output_area:
                output_area.clear_output()
                if movies_input.value.strip():
                    results = self.search(
                        query= movies_input.value, 
                    
                    )
                    display(results)

        movies_input.observe(on_type, names='value')
        display(widgets.VBox([movies_input, output_area]))
