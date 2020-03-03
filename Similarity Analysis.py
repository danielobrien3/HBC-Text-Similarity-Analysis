#!/usr/bin/env python
# coding: utf-8

# In[2]:


# In an attempt to make this notebook as organized (re: readable) as possible, I will explain its structure here. 
## The first cell is dedicated to initializing all dependencies used. 
## Each cell after is then dedicated to defining each function in the module
### Finally, the last cell is this project's driver. That is where we will be putting the pieces of the puzzle together.


# In[3]:


# Imports. 'Requests' for https requests. 'BeautifulSoup' for html scraping. 'Pandas' for data analysis. 
# 'sklearn' for similarity functions, such as word counter and cosine similarity. 'gensim' for Doc2Vec.
# 'nltk' for pre-processing main text. 're' for regex. 'scipy' for spacial cosine. 

import requests
import argparse
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import doc2vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from scipy import spatial
import gensim
from collections import Counter
import copy
from pathlib import Path
import numpy as np
import time



# Load Google's pre-trained Word2Vec model.
data_folder = Path("data/")
file = data_folder / "GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True, limit=500000)
index2word_set = set(model.wv.index2word)

# This can get initialized up here, as it will be constant throughout. 
count_vectorizer = CountVectorizer(stop_words='english')


# In[4]:


# Corpus class. This will hold all the article objects in our corpus and allow us to execute some corpus-wide methods.

class Corpus:
    def __init__(self, main_article):
        self.main_article = WikiArticle(main_article)
        self.articles = []
        self.corpus_stopwords = []
        self.D2Vmodel = None
        
    def fill_corpus(self, size, mode):
        """Function: Fills corpus by getting related articles, starting with the main article and
           using the other articles that are found until the corpus meets the set size parameter.  
           ============================================================================
           Parameters
           ----------
           Desired size of final corpus.

           Returns
           ----------
           No return. Corpus articles are stored in self.articles"""
        if(mode == "all_random"):
            while len(self.articles) < size:
                art = WikiArticle("https://en.wikipedia.org/wiki/Special:Random")
                self.articles.append(art)
                
        if(mode == "all_related"):
            # Start filling corpus with artiles related to main article
            self.articles = self.main_article.get_related()

            # Keep track of current article using article counter
            article_counter = 0

            while len(self.articles) < size:
                related_articles = self.articles[article_counter].get_related()
                if(related_articles != False):
                    self.articles.extend(self.articles[article_counter].get_related())
                article_counter +=1
                
        if(mode=="50_50"):
            #Start filling corpus with articles related to main article
            article_counter = 0
            self.articles = self.main_article.get_related()
            while len(self.articles) < size/2:
                related_articles = self.articles[article_counter].get_related()
                if(related_articles != False):
                    self.articles.extend(self.articles[article_counter].get_related())
                article_counter +=1
                
            while len(self.articles) < size:
                art = WikiArticle("https://en.wikipedia.org/wiki/Special:Random")
                self.articles.append(art)
            
            
        
    def build_model(self):
        """Function: Analyzes all corpus against corpus main article and returns results in pandas dataframe.
           ============================================================================
           Parameters
           ----------
           None 

           Returns
           ----------
           No return. Builds and trains doc2vec model and stores it in corpus object"""
        corpus = []
        self.main_article.get_main_text()
        corpus.append(gensim.models.doc2vec.TaggedDocument(words=self.main_article.main_text, tags=[0]))
        tag_counter = 0
        for article in self.articles:
            article.get_main_text()
            corpus.append(gensim.models.doc2vec.TaggedDocument(words=article.main_text, tags=[tag_counter]))
            tag_counter += 1
            
        D2Vmodel = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=10)
        D2Vmodel.build_vocab(corpus)
        D2Vmodel.train(corpus, total_examples=D2Vmodel.corpus_count, epochs=20)
        self.D2Vmodel = D2Vmodel            
        
    def similarity_analysis(self, smart):
        """Function: Analyzes all corpus against corpus main article and returns results in pandas dataframe.
           ============================================================================
           Parameters
           ----------
           <boolean> Determines whether or not multi-tiered analysis is used. 

           Returns
           ----------
           Returns pandas dataframe with analysis results"""
        
        ## Pre-process main article
        self.main_article.get_secondary_titles()
        self.main_article.get_main_text()
        self.main_article.get_word_frequency()
        
        ## Build and train model
        self.build_model()
        
        ## Initialize Pandas Dataframe to store results
        index = []
        for article in self.articles:
            if(article.main_title not in index):
                index.append(article.main_title)

        columns = ["Main Title Tier", "Secondary Title Tier", "Main Analysis"]
        results = pd.DataFrame(index=index, columns=columns)

        ## Actually do the analysis
        ndx = 1 ## index for d2v tagging purposes
        for article in self.articles:
            results.loc[article.main_title] = self.main_article.similarity_analysis(article, smart, ndx, self.D2Vmodel)
            ndx = ndx + 1
        
        return results
            
    
    def new_article(self, url):
        """Function: add a specific wiki article to the corpus.
           ============================================================================
           Parameters
           ----------
           (Wiki) Article url

           Returns
           ----------
           Returns index of new article."""
        self.articles.append(WikiArticle(url))
        return len(self.articles) - 1
    
    def filter_corpus_by_frequency(self, *args):
        """Function: Finds a variable amount (default 3) of the most frequent words in the corpus. 
           These words are then removed from all of the article.word_frequency[] dictionaries. 
           ============================================================================
           Parameters
           ----------
           (Optional) <int> Number of stop words to get. Default is 3. 

           Returns
           ----------
           No return. Filters most frequent words in corpus out of all articles word_frequency dictionaries."""

        
        ## Lets loop through the articles in the corpus, get each articles word_frequency count, and merge them into a common dictionary.
        # ...this will be fun
        total_frequency = {}
        for article in self.articles:
            total_frequency = mergeDict(total_frequency, article.get_word_frequency())
        total_frequency = Counter(total_frequency)
        ## Okay now that's done, let's get the most frequently found words in the corpus (aka our new corpus stop words).
        
        # Check if optional paramater was passed. 
        if(len(args) == 1):
            count = args[0]
        else:
            count = 3

        # Get 3 most frequently found words and store them in corpus_stopwords list. 
        self.corpus_stopwords = [item[0] for item in total_frequency.most_common(count)]
        
        # Loop through all articles in corpus, filtering out the newly obtained corpus stop words.
        for article in self.articles:
            article.filter_corpus_stopwords(self.corpus_stopwords)
        
        
# Class agnostic function to help merging word_frequency dicts
def mergeDict(dict1, dict2):
    ''' Merge dictionaries and keep values of common keys in list'''
    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = (value + dict1[key])
    return Counter(dict3)


# In[5]:


# wikiArticle class. Named 'wikiArticle' for lack of inspiration. Will hold all relevant data on an article. 

class WikiArticle:
    def __init__(self, url):
        self.url = url
        self.soup = BeautifulSoup(requests.get(self.url).text, "html")
        self.main_title = self.soup.find_all("h1")[0].get_text()
        self.secondary_titles = ""
        self.main_text = ""
        self.word_frequency = {}
        
    
    def similarity_analysis(self, article, smart, ndx, D2Vmodel):
        """
        Function: To be used in Corpus class to perform analysis between main article and comparison article
        ============================================================================
           Parameters
           ----------
           Article to be analyzed against main article.

           Returns
           ----------
           Returns Pandas series to be used in forming a dataframe result."""
        ## Check main title tier
        if(smart == True):
            a_vec = self.w2v_vector(self.main_title);
            b_vec = article.w2v_vector(article.main_title);
            main_title_comparison = vec_cosine_analysis(a_vec, b_vec)
            if is_over_threshold(main_title_comparison):
                main_title_tier = True
            else:
                return pd.Series({'Main Title Tier': main_title_comparison, 'Secondary Title Tier': None, 'Main Analysis': None})

            ## Check sub title tier
            article.get_secondary_titles()
            a_vec = self.w2v_vector(self.secondary_titles)
            b_vec = article.w2v_vector(article.secondary_titles)
            secondary_title_comparison = vec_cosine_analysis(a_vec, b_vec)
            if is_over_threshold(secondary_title_comparison):
                secondary_title_tier = True
            else:
                return pd.Series({'Main Title Tier': main_title_comparison, 'Secondary Title Tier': secondary_title_comparison, 'Main Analysis': None})
        else: 
            main_title_comparison = None;
            secondary_title_comparison = None;
        
        ## Since we got this far we should get the main text and wrod count of the article
        article.get_main_text()
        article.get_word_frequency()
        
        ## And now perform the main analysis 
        a_vec = D2Vmodel.infer_vector(self.main_text)
        b_vec = D2Vmodel.infer_vector(article.main_text)
        analysis = vec_cosine_analysis(a_vec, b_vec)
        return pd.Series({'Main Title Tier': main_title_comparison, 'Secondary Title Tier': secondary_title_comparison, 'Main Analysis':analysis})
        
            
    
    def get_secondary_titles(self):
        # Check length to make sure secondary_titles list hasn't already been filled. Don't want duplicate data messing us up. 
        if(len(self.secondary_titles) == 0):
            for secondary_title in self.soup.find_all("h2"):
                self.secondary_titles += " " + secondary_title.get_text()
    
    def w2v_vector(self, text):
        """
        Function: To get word2vec average vector of provided text
        ============================================================================
           Parameters
           ----------
           Text to be word2vec'd

           Returns
           ----------
           Word2Vec average vector of provided text."""
        num_features = 300;
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0

        for word in text:
            if word in index2word_set:
                nwords = nwords+1
                featureVec = np.add(featureVec, model[word])

        if nwords>0:
            featureVec = np.divide(featureVec, nwords)
        return featureVec
        
        
                
    def get_main_text(self):
        """
        Function: self.main_text set to <string> pre-processed main text of article.
        ============================================================================
           Parameters
           ----------
           Takes no parameters.

           Returns
           ----------
           Returns nothing."""
        
        # Gets text from the article
        paragraphs = self.soup.find_all("p")
        article_text = ""
        for p in paragraphs:
            article_text = article_text + " " + p.text
        
        # Prepares text for analysis.
        self.main_text = self.pre_process(article_text)
    
    def get_word_frequency(self):
        """ 
        Function: gets word frequencies for article and stores in self.word_frequency dictionary. 
        ===========================================
           Parameters
           ----------
           Takes no paramaters

           Returns
           ----------
           Returns word frequency. Result is stored in self.word_frequency"""
        
        self.word_frequency = Counter(self.main_text)
        return self.word_frequency
        
        
    def filter_corpus_stopwords(self, corpus_stop_words):
        """ 
        Function: removes all occurences of the most frequent words in the corpus from self.word_frequency. This function should only be called from within a corpus class method.
        ===========================================
        Parameters
        ----------
        <list> corpus stop words

        Returns
        ----------
        No return. self.word_frequency"""
        
        filtered_text = Counter({})
        for k, v in self.word_frequency.items():
            if not k in corpus_stop_words:
                filtered_text[k] = v
                
        self.word_frequency = filtered_text
    
    
    def pre_process(self, text):
        """
        Function: pre-processes text to prepare for analysis. 
        =====================================================
           Parameters
           ----------
           Takes <string> text to be pre-processed.

           Returns
           ----------
           Returns <dict> Doc2Vec of pre-processed text."""
        
        # Cleaing the text
        processed_article = text.lower()
        
        # Preparing the dataset
        all_words = word_tokenize(processed_article)
        processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
        processed_article = re.sub(r'\s+', ' ', processed_article)
        
        # Removing Stop Words
        processed_text = []
        for w in all_words:
            if not w in stopwords.words('english') and not w in string.punctuation:
                processed_text.append(w)
        
        return processed_text

    
    def get_related(self):
        """ Function: Get list of related articles
            =====================================================
                Parameters
                ----------
                This function takes no paramater 
                
                Returns
                -------
                If related articles exist, they are returned in a list. Else returns false. 
               
               """
        if(self.soup.find(id="See_also") is None):
            return False
        else:
            related_list = self.soup.find(id="See_also").parent.find_next('ul').findChildren('li')

            articles = []
            for item in related_list:
                link = item.findChild('a')
                articles.append(WikiArticle("https://en.wikipedia.org"+link.get('href')))

            return articles
            


# In[6]:


def jaccard_analysis(article_one, article_two):
    
    """Parameters
       ----------
       Right now this function takes two strings as its parameters (article_one, article_two). In the future, it should take 
       WikiArticle instances to allow multiple sub-headers to be analyzed together. 
       
       Returns
       --------
       Jaccard Similarity Percentage."""
    
    a = set(article_one.split(" "))
    b = set(article_two.split(" "))
    comparison = a.intersection(b)
    return float(len(comparison)) / (len(a) + len(b) - len(comparison))


# In[7]:


def cosine_analysis(article_one_frequency, article_two_frequency):

    # convert to word-vectors
    words  = list(article_one_frequency.keys() | article_two_frequency.keys())
    a_vect = [article_one_frequency.get(word, 0) for word in words]        
    b_vect = [article_two_frequency.get(word, 0) for word in words]       

    # find cosine
    len_a  = sum(av*av for av in a_vect) ** 0.5             
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5             
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    
    cosine = dot / (len_a * len_b)
    
    # return cosine
    return cosine

def vec_cosine_analysis(a_vect, b_vect):
    # find cosine
    len_a  = sum(av*av for av in a_vect) ** 0.5             
    len_b  = sum(bv*bv for bv in b_vect) ** 0.5             
    dot    = sum(av*bv for av,bv in zip(a_vect, b_vect))    
    cosine = dot / (len_a * len_b)
    
    # return cosine
    return cosine
    


# In[8]:


def is_over_threshold(similarity, *args):
    
    """Parameters
       ----------
       similarity (float): similarity value that will be checked against threshold.
       threshold (float): Optional paramater to provide value for threshold. Must be passed as "threshold = (value)". Default is 50.
    
       Returns
       ----------
       Boolean value. True if threshold limit is met or exceeded, else False."""
    
    if(len(args) == 1):
        threshold = args[0]
    else:
        threshold = .5
    return (similarity >= threshold)


# In[9]:


### Driver ###
##          ##
# ========== #/

# Initiate corpus with article of focus
if __name__ == "__main__":
    start = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int)
    parser.add_argument("fill_type")
    args = parser.parse_args()
    corpus = Corpus("https://en.wikipedia.org/wiki/IBM_mainframe")
    corpus.fill_corpus(args.size, args.fill_type)
    corpus.filter_corpus_by_frequency()

    smartResults = corpus.similarity_analysis(True)
    dumbResults = corpus.similarity_analysis(False)
    end = time.time()
    print("Took " + str(end - start) + "s to run")
    fileOne = "multitiered_size_"  + str(args.size) + "_corpus_" +args.fill_type +".csv"
    fileTwo = "regular_size_"  + str(args.size) + "_corpus_" +args.fill_type +".csv"
    smartResults.to_csv(fileOne)
    dumbResults.to_csv(fileTwo)


# In[ ]:





# In[9]:





# In[ ]:





# In[ ]:





# In[ ]:




