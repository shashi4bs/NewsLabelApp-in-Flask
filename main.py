from lib import *
from features import *


#dataset "train.csv" should be present in the present working directory 
data = pd.read_csv('./train_data.csv');

#data.head()

#uncomment to get desired dataset size
#size = 2000
#data = data[:size]
#uncomment to select a fraction of dataset
data = data.sample(frac=0.1).reset_index(drop=True)


categories = data.CATEGORY.factorize()

#uncomment to get data visualisation
#plt.hist(categories[0],categories[1].shape[0])


#print("RAW Text: ",data['TITLE'][1])
#text normalisation
print("Processing Headlines...")

for line,i in zip(data['TITLE'],range(data['TITLE'].shape[0])):
    data.loc[i,('TITLE')] = normalise_text(line)

#print("Normalized Text: ",data['TITLE'][1])


cv_matrix = countVectorizer(data)
categories = categories[0]

#splitting data into training and test set
training_data, testing_data, training_op, test_op = split_data(cv_matrix,categories)

dtc_model = DecisionTreeClassifier(min_samples_split=8,max_features='auto',criterion='entropy',random_state=10)

print("Training Model...")
dtc_model.fit(training_data,training_op)
print("Training Complete.")

print("Model Accuracy on Training Data: ",dtc_model.score(training_data,training_op))

print("Model Accuracy on Training Data: ",dtc_model.score(testing_data,test_op))
