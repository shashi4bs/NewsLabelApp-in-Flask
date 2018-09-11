from lib import *
word_tokenizer = nltk.word_tokenize
stop_words = nltk.corpus.stopwords.words('english')


def normalise_text(data):
    # lower case and remove special characters\whitespaces
    data = re.sub(r'[^a-zA-Z0-9\s]', '', data, re.I)
    data = data.lower()
    data = data.strip()
    tokens = word_tokenizer(data)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    data = ' '.join(filtered_tokens)
    return data
    
def tfidfTransformer(data_, tv=None):
	tv = TfidfTransformer()
	tv_matrix = tv.fit_transform(data_)
	return tv_matrix, tv
	
def countVectorizer(data, cv=None):
	cv = CountVectorizer()
	cv_matrix = cv.fit_transform(data.TITLE)
	return cv_matrix,cv
	
def split_data(cv_matrix,categories):
    size = cv_matrix.shape[0]
    training_data = cv_matrix[:int(size*0.80)]
    testing_data = cv_matrix[int(size*0.80):size]
    training_op = categories[:int(size*0.80)]
    testing_op = categories[int(size*0.80):size]
    return training_data,testing_data,training_op,testing_op
    
def remove_tag(lists,headlines):
    tag = re.compile("<.*?>")
    for l in lists:
        ret = re.sub(tag," ",str(l))
        ret = ret.strip()
        if(len(ret)):
            headlines.append(ret)
    return headlines

def extract_hedlines(url):
	try:
		page = requests.get(url)
		content = BeautifulSoup(page.content,'html.parser')
		lists = content.find_all('p')
		heading = content.find_all('h1')
		span = content.find_all('span')
		headlines = []
		headlines = remove_tag(heading,headlines)
		headlines = remove_tag(lists,headlines)
		headlines = remove_tag(span,headlines)
		return headlines
	except:
		print("Error Retrieving data: ",url)
