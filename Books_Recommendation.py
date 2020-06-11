#Importing library
import pandas as pd
#import Dataset 
books = pd.read_csv(r"D:\Python\New folder\books.csv",encoding="latin-1")
books.shape #shape
books.columns
books.Publisher
books.Ratings

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(books.Publisher)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #5000,5822

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of book name to index number 
books_index = pd.Series(books.index,index=books['Title']).drop_duplicates()


books_index["Los Renglones Torcidos De Dios"]

def get_books_recommendations(Title,topN):
    
   
    #topN = 10
    # Getting the book index using its title 
   books_id = books_index[Title]
    
    # Getting the pair wise similarity score for all the book's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[books_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar book's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the book index 
    books_idx  =  [i[0] for i in cosine_scores_10]
    books_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar books and scores
    books_similar_show = pd.DataFrame(columns=["Title","Score"])
    books_similar_show["Title"] = books.loc[anime_idx,"Title"]
    books_similar_show["Score"] = books_scores
    books_similar_show.reset_index(inplace=True)  
    books_similar_show.drop(["index"],axis=1,inplace=True)
    print (books_similar_show)
    #return (books_similar_show)

    
# Enter your anime and number of book's to be recommended 
get_books_recommendations("Pnin",topN=15)


#### Author Based####
tfidf_matrix = tfidf.fit_transform(books.Publisher)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #5000,5822

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of book name to index number 
books_index = pd.Series(books.index,index=books['Title']).drop_duplicates()


books_index["Los Renglones Torcidos De Dios"]

def get_books_recommendations(Title,topN):
    
   
    #topN = 10
    # Getting the book index using its title 
   books_id = books_index[Title]
    
    # Getting the pair wise similarity score for all the book's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[books_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar book's 
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the book index 
    books_idx  =  [i[0] for i in cosine_scores_10]
    books_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar books and scores
    books_similar_show = pd.DataFrame(columns=["Title","Score"])
    books_similar_show["Title"] = books.loc[anime_idx,"Title"]
    books_similar_show["Score"] = books_scores
    books_similar_show.reset_index(inplace=True)  
    books_similar_show.drop(["index"],axis=1,inplace=True)
    print (books_similar_show)
    #return (books_similar_show)

    
# Enter your anime and number of book's to be recommended 
get_books_recommendations("Pnin",topN=15)