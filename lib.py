import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re

nltk.download('punkt')
nltk.downnload('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
