from lib import *
from features import *


#dataset "train.csv" should be present in the present working directory 
data = pd.read_csv('./train_data.csv');

#data.head()

#uncomment to get desired dataset size
#size = 2000
#data = data[:size]

categories = data.CATEGORY.factorize()

#uncomment to get data visualisation
#plt.hist(categories[0],categories[1].shape[0])


#print("RAW Text: ",data['TITLE'][1])
#text normalisation
for line,i in zip(data['TITLE'],range(data['TITLE'].shape[0])):
    data['TITLE'][i] = normalise_text(line)
#print("Normalized Text: ",data['TITLE'][1])

