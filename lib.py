import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import os
import pickle

nltk.download('punkt')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from bs4 import BeautifulSoup
import requests
