
from lib import *
from features import *

os.system('clear')
# In[2]:
print('Fetching News:')

url = ["https://www.bbc.co.uk/","https://in.yahoo.com/?p=us",'https://gadgets.ndtv.com/news',"https://timesofindia.indiatimes.com/business/india-business/met-finance-minister-before-leaving-the-country-vijay-mallya/articleshow/65785080.cms",'https://news.google.com/?hl=en-IN&gl=IN&ceid=IN:en']



categories = ['Medical','Entertainment','Business','Tech']

headlines = []
for i in range(len(url)):
    headlines.append(extract_hedlines(url[i]))

for i in range(len(headlines)):
    for lines, j in zip(headlines[i], range(len(headlines[i]))):
    	headlines[i][j] = normalise_text(lines)


# # converting lists of headlines to dataframe

testing_headlines ={}
for i in range(len(headlines)):
    testing_headlines[url[i]] = pd.DataFrame({"TITLE":headlines[i]})




testing_headlines.keys()


# # load trained model, count_vectorizer, tf-idf

# In[9]:

print("Loading Stored Models..")
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl','rb'))
tv = pickle.load(open('tv.pkl','rb'))



print('processing Headlines...')
processed_data = {}
transformer = Transformer(cv, tv)
for link in testing_headlines.keys():
    processed_data[link] = transformer.transform(testing_headlines[link]['TITLE'])

# In[11]:


prediction = {}
for link in testing_headlines.keys():
    prediction[link] = model.predict(processed_data[link])


# In[14]:


for link in testing_headlines.keys():
    print(link)
    for i in range(len(testing_headlines[link])):
        if(len(testing_headlines[link]['TITLE'][i])>40 and len(testing_headlines[link]['TITLE'][i])<100):
            print(testing_headlines[link]['TITLE'][i]," :   ",categories[prediction[link][i]])

