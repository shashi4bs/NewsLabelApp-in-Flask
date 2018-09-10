from lib import *
from features import *


#dataset "train.csv" should be present in the present working directory 
data = pd.read_csv('./train_data.csv');

#data.head()

#uncomment to get desired dataset size
#size = 2000
#data = data[:size]
#uncomment to select a fraction of dataset
data = data.sample(frac=0.01).reset_index(drop=True)


categories = data.CATEGORY.factorize()

#uncomment to get data visualisation
#plt.hist(categories[0],categories[1].shape[0])


#print("RAW Text: ",data['TITLE'][1])
#text normalisation
print("Processing Headlines...")

for line,i in zip(data['TITLE'],range(data['TITLE'].shape[0])):
    data.loc[i,('TITLE')] = normalise_text(line)

#print("Normalized Text: ",data['TITLE'][1])


cv_matrix, cv = countVectorizer(data)
#SAVE WORD VECTOR
pickle.dump(cv, open("cv.pkl","wb"))

tv_matrix, tv = tfidfTransformer(cv_matrix)
#Save tf-idf
pickle.dump(tv, open("tv.pkl","wb"))

categories = categories[0]

#splitting data into training and test set
training_data, testing_data, training_op, test_op = split_data(tv_matrix,categories)

rfc_model = RandomForestClassifier(min_samples_split=4,criterion='entropy',random_state=10)

print("Training Model...")
rfc_model.fit(training_data,training_op)
print("Training Complete.")

#SAVE MODEL
pickle.dump(rfc_model, open("rfc_model.pkl", "wb"))
print("Model Saved.")

print("Model Accuracy on Training Data: ",rfc_model.score(training_data,training_op))

print("Model Accuracy on Testing Data: ",rfc_model.score(testing_data,test_op))
