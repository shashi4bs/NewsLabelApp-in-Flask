
# coding: utf-8

# In[1]:


from lib import *
from features import *

os.system('clear')
# In[2]:
print('Fetching News:')

url = ["https://www.bbc.co.uk/","https://in.yahoo.com/?p=us",'https://gadgets.ndtv.com/news',"https://timesofindia.indiatimes.com/business/india-business/met-finance-minister-before-leaving-the-country-vijay-mallya/articleshow/65785080.cms",'https://news.google.com/?hl=en-IN&gl=IN&ceid=IN:en']


# In[3]:


categories = ['Medical','Entertainment','Business','Tech']


# In[4]:


headlines = []
for i in range(len(url)):
    headlines.append(extract_hedlines(url[i]))


# In[5]:

#considering it to be a specific page for an article 
# In[6]:


for i in range(len(headlines)):
    for lines, j in zip(headlines[i], range(len(headlines[i]))):
    	headlines[i][j] = normalise_text(lines)


# # converting lists of headlines to dataframe

# In[7]:


testing_headlines ={}
for i in range(len(headlines)):
    testing_headlines[url[i]] = pd.DataFrame({"TITLE":headlines[i]})


# In[8]:


testing_headlines.keys()


# # load trained model, count_vectorizer, tf-idf

# In[9]:

print("Loading Stored Models..")
model = pickle.load(open('rfc_model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl','rb'))
tv = pickle.load(open('tv.pkl','rb'))


# In[10]:

print('processing Headlines...')
cv_matrix = {}
tv_matrix = {}
for link in testing_headlines.keys():
    cv_matrix[link] = cv.transform(testing_headlines[link]['TITLE']).toarray()
    tv_matrix[link] = tv.transform(cv_matrix[link]).toarray()


# In[11]:


prediction = {}
for link in testing_headlines.keys():
    prediction[link] = model.predict(tv_matrix[link])


# In[14]:


for link in testing_headlines.keys():
    print(link)
    for i in range(len(testing_headlines[link])):
        if(len(testing_headlines[link]['TITLE'][i])>25 and len(testing_headlines[link]['TITLE'][i])<100):
            print(testing_headlines[link]['TITLE'][i]," :   ",categories[prediction[link][i]])

