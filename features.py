from lib import *
word_tokenizer = nltk.word_tokenize
stop_words = nltk.corpus.stopwords.words('english')

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, cv=None, tv=None):
        if not cv:
            self.cv = CountVectorizer()
        else:
            self.cv = cv
        if not tv:
            self.tv = TfidfTransformer(norm='l2',use_idf=True)
        else:
            self.tv = tv

    def fit(self):
        return self

    def fit_transform(self, data):
        self.cv_matrix = self.cv.fit_transform(data)
        self.tv_matrix = self.tv.fit_transform(self.cv_matrix)
        return self.cv, self.tv,self.tv_matrix
    
    def transform(self, data):
        return self.tv.transform(self.cv.transform(data))
        
def normalise_text(data):
    # lower case and remove special characters\whitespaces
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data, re.I)
    data = data.lower()
    data = data.strip()
    tokens = word_tokenizer(data)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    data = ' '.join(filtered_tokens)
    return data

def remove_tag(lists,headlines):
    tag = re.compile("<.*?>")
    for l in lists:
        ret = re.sub(tag," ",str(l))
        ret = ret.strip()
        if(len(ret)):
            headlines.append(ret)
    return headlines

def extract_hedlines(url):
	page = requests.get(url)
	content = BeautifulSoup(page.content,'html.parser')
	lists = content.find_all('p')
	heading = content.find_all('h1')
	span = content.find_all('span')
	links = content.find_all('a',class_='VDXfz')
	for i in range(len(links)):
		links[i] = links[i].text
	headlines = []
	headlines = remove_tag(heading,headlines)
	headlines = remove_tag(lists,headlines)
	headlines = remove_tag(span,headlines)
	headlines = remove_tag(links,headlines)
	return headlines
	
