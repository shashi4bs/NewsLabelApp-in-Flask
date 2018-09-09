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
    
def tfidfVectorizer(data):
	tv = TfidfVectorizer(min_df=0.,max_df=1.,use_idf=True)
	tv_matrix = tv.fit_transform(data['TITLE'])
	tv_matrix = tv_matrix.toarray()
	vocab = tv.get_feature_names()
	return (vocab,tv_matrix)
	
def countVectorizer(data):
	cv = CountVectorizer(min_df=0.,max_df=1.)
	cv_matrix = cv.fit_transform(data['TITLE'])
	return cv_matrix.toarray()
	
def split_data(cv_matrix,categories):
    size = cv_matrix.shape[0]
    training_data = cv_matrix[:int(size*0.80)]
    testing_data = cv_matrix[int(size*0.80):size]
    training_op = categories[:int(size*0.80)]
    testing_op = categories[int(size*0.80):size]
    return training_data,testing_data,training_op,testing_op 
