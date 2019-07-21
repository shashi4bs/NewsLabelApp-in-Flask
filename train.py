from lib import *
from features import *

PATH = os.path.dirname(__file__)
#RAW FILE
#FILENAME = 'train_data.csv'

#dataset "train.csv" should be present in the present working directory 

'''
remove of comment line 30 and 31 while using NormalizedText.csv
NormalizedText.csv is normalized file for train_data.csv
'''
FILENAME = 'NormalizedText.csv'
data = pd.read_csv(PATH+FILENAME);
print(PATH+FILENAME)

#uncomment to get desired dataset size
#size = 2000
#data = data[:size]
#uncomment to select a fraction of dataset
#data = data.sample(frac=0.4).reset_index(drop=True)

data = data.dropna()

categories = data.CATEGORY.factorize()
#uncomment to get data visualisation
#plt.hist(categories[0],categories[1].shape[0])


#print("RAW Text: ",data['TITLE'][1])
#text normalisation
print("Processing Headlines...")

'''
for line,i in zip(data['TITLE'],range(data['TITLE'].shape[0])):
    data.loc[i,('TITLE')] = normalise_text(line)
'''
#print("Normalized Text: ",data['TITLE'][1])
transformer = Transformer()
cv, tv, processed_data = transformer.fit_transform(data.TITLE)

#SAVE WORD VECTOR
pickle.dump(cv, open("cv.pkl","wb"))
#Save tf-idf
pickle.dump(tv, open("tv.pkl","wb"))

categories = categories[0]

#splitting data into training and test set
X_train, X_test, y_train, y_test = train_test_split(processed_data, categories)

model = BernoulliNB(alpha=1.0, binarize=0.0)
print(model)
print("Training Model...")
model.fit(X_train, y_train)
print("Training Complete.")

#SAVE MODEL
pickle.dump(model, open("model.pkl", "wb"))
print("Model Saved.")

print("Model Accuracy on Training Data: {:.2f}%".format(100 * model.score(X_train,y_train)))

print("Model Accuracy on Testing Data: {:.2f}%".format(100 * model.score(X_test,y_test)))
