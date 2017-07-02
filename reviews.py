from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


#open datasets
imdbFile = "C:\\Users\\Bradley\\Desktop\\machine learning\Reviews\\imdb_labelled.txt";
yelpFile = "C:\\Users\\Bradley\\Desktop\\machine learning\\Reviews\\yelp_labelled.txt";
amazonFile = "C:\\Users\\Bradley\\Desktop\\machine learning\\Reviews\\amazon_cells_labelled.txt";

with open(imdbFile, "r") as text_file: lines = text_file.read().split('\n')
with open(yelpFile, "r") as text_file: lines += text_file.read().split('\n')
with open(amazonFile,"r") as text_file: lines += text_file.read().split('\n')

#remove corrupted data
lines = [line.split("\t") for line in lines if len(line.split("\t")) == 2 and line.split("\t")]


#put sentences and labels in training set ["sentence", label (0/1)]
train_sentences = [line[0] for line in lines]
train_labels =  [int(line[1]) for line in lines]

count_vectorizer = CountVectorizer(binary='true')
train_sentences = count_vectorizer.fit_transform(train_sentences)
classifier = BernoulliNB().fit(train_sentences, train_labels)

reviewInput = raw_input("Enter a review: ")

#use sample review program will output if it was a good review or a bad one
result = classifier.predict(count_vectorizer.transform([reviewInput]))


#1 good 0 bad
if result == 1:
   print ("This is a good review")
else:
   print("This is a bad review")   

