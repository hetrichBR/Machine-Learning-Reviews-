from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


#open datasets
imdbFile = "C:\\Users\\bhetrich\Desktop\\ML\\Classify\\reviews\\sentiment labelled sentences\\sentiment labelled sentences\\imdb_labelled.txt";

with open(imdbFile, "r") as text_file: lines = text_file.read().split('\n')

#remove corrupted data
lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")]

#use kmeans to cluster
train_documents = [line[0] for line in lines]

#convert each review to its tfid Term Frequency Inverse Doc
tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
train_documents = tfidf_vectorizer.fit_transform(train_documents)

#define # of clusters and max iteration time
km = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 1, verbose = True)
km.fit(train_documents)

#print reviews of group 2 experiment with k categories 3 = 0,1,2
count = 0
for i in range(len(lines)):
    
	if km.labels_[i] == 2:
	   print(lines[i])
	   count += 1