import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from nltk import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,accuracy_score
from xgboost import XGBClassifier 
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore", category=DeprecationWarning)


tfidf_matrix = ''
tfidf = ''
train = ''
tokenized_tweet = ''
df_tfidf = ''
Log_Reg = ''
XGB = ''
DecTree = ''
fixed_label = 'fixed'
log_acc = 0
xgb_acc = 0
dec_acc = 0


def getDataset(dir):
    return pd.read_csv(dir,encoding='latin1')

def createDataset(name,text):
    nme = [text]
    # dictionary of lists 
    dict = {'text': nme}     
    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv(name+'.csv',index=False)

def fixLabels(dataset):
   
    return dataset['label'].replace(
    {   'neutral'       :0,
        'positive'     :1,
        'negative'   :2
    })
    

def remove_pattern(text,pattern):
    
    # re.findall() finds the pattern i.e @user and puts it in a list for further task
    r = re.findall(pattern,text)
    
    # re.sub() removes @user from the sentences in the dataset
    for i in r:
        text = re.sub(i,"",text)
    
    return text

def removePatternDataset(dataset):
    return  np.vectorize(remove_pattern)(dataset['text'], "@[\w]*")

def removeSpecialChars(dataset,fixed_label):
    return  dataset[fixed_label].str.replace("[^a-zA-Z#]", " ", regex=True)
    
def lambdaDataset(dataset,fixed_label):
    return dataset[fixed_label].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

def tokenize(dataset,fixed_label):
    return dataset[fixed_label].apply(lambda x: x.split())

def tokenizePortStemmer(tokenized_tweet):
    ps = PorterStemmer()

    tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

    return tokenized_tweet

def applyTokenize(tokenized_tweet,dataset,fixed_label):
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

    dataset[fixed_label] = tokenized_tweet
    return dataset
    
def TFIDF(dataset,fixed_label):
    
    tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=20000,stop_words='english')

    tfidf_matrix=tfidf.fit_transform(dataset[fixed_label])

    df_tfidf = pd.DataFrame(tfidf_matrix.todense())
    
    return df_tfidf

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    text = " ".join(lst_text)
    return text

def textCleaner(df = None , src = 'comment_text' ,dst = 'text_clean',stop_words = 'english'):
    
    if df is None:
        raise TypeError("Data Frame cannot be type 'None'")
    
    try:
        lst_stopwords = nltk.corpus.stopwords.words(stop_words)
    except:
        raise Exception( "'" + stop_words +"'"+" is not a valid type.")
    df[dst] = df[src].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))


nltk.download('stopwords')
nltk.download('wordnet')

def cleanDataset(dataset,fixed_label):
    print('remove pattern')
    dataset[fixed_label] = removePatternDataset(dataset)
    print (train)
    
    
    #remote specials
    print('remove specials')
    dataset[fixed_label] = removeSpecialChars(dataset,fixed_label)
    
    #apply lambda
    print('apply lambda')
    dataset[fixed_label] = lambdaDataset(dataset,fixed_label)
    
    return dataset

def fitModels(x_train_tfidf,y_train_tfidf,x_valid_tfidf,y_valid_tfidf):
    global Log_Reg,XGB,DecTree,fixed_label,log_acc,xgb_acc,dec_acc
    
    #-----------LOGISTIC-------
    Log_Reg = LogisticRegression(random_state=47,solver='newton-cg')
    Log_Reg.fit(x_train_tfidf,y_train_tfidf)
    prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)
    print(prediction_tfidf)
    prediction_int = np.argmax(prediction_tfidf,axis = 1)
    print('Logistic Accuracy : ')
    log_acc = accuracy_score(y_valid_tfidf,prediction_int)
    print(accuracy_score(y_valid_tfidf,prediction_int))
    
    #-----------XGBOOST-------
    
    XGB=XGBClassifier(random_state=29,learning_rate=0.9)
    XGB.fit(x_train_tfidf, y_train_tfidf)
    xgb_tfidf=XGB.predict_proba(x_valid_tfidf)
    prediction_int = np.argmax(xgb_tfidf,axis = 1)
    print('XGBoost Accuracy : ')
    xgb_acc = accuracy_score(y_valid_tfidf,prediction_int)
    print(accuracy_score(y_valid_tfidf,prediction_int))
    
    
    #-----------Decision Tree---------
    
    DecTree = DecisionTreeClassifier(criterion='entropy', random_state=1)
    DecTree.fit(x_train_tfidf,y_train_tfidf)
    dct_tfidf = DecTree.predict(x_valid_tfidf)
    print('Decision Accuracy : ')
    dec_acc = accuracy_score(dct_tfidf,y_valid_tfidf)
    print(accuracy_score(dct_tfidf,y_valid_tfidf))

def predict(textToPredict,csvName,label):
    global tfidf,Log_Reg,XGB,DecTree,fixed_label
    
    createDataset(csvName,textToPredict)
    
    testData = getDataset(csvName+'.csv')
    
    testData = cleanDataset(testData,fixed_label)
    
    test_tokenize = tokenize(testData,fixed_label)
    test_tokenize = tokenizePortStemmer(test_tokenize)
    
    testData = applyTokenize(test_tokenize,testData,fixed_label)
    
    tfidf_matrix=  tfidf.transform(testData[label])

    df_tfidf = pd.DataFrame(tfidf_matrix.todense())
    
    prediction_log = Log_Reg.predict_proba(df_tfidf)
    prediction_xgb = XGB.predict_proba(df_tfidf)
    prediction_dec = DecTree.predict(df_tfidf)
    
    print("Logistic Regression")
    print(prediction_log)
    print("XGB")
    print(prediction_xgb)
    print("Decision Tree")
    print(prediction_dec)
    
    return prediction_log,prediction_xgb,prediction_dec
    
    
    
# Defining main function
def TrainModels():
    global tfidf,Log_Reg,XGB,DecTree,fixed_label,log_acc,xgb_acc,dec_acc
    
    if Log_Reg == '' or DecTree == '' or XGB == '':
        print("hey there")
        
        #Get Dataset
        train = getDataset('all-data4.csv')
        
        #Label Fixing
        train['label'] = fixLabels(train)
        print(train)
        
        #remove pattern
        print('remove pattern')
        train[fixed_label] = removePatternDataset(train)
        print (train)
        
        
        #remote specials
        print('remove specials')
        train[fixed_label] = removeSpecialChars(train,fixed_label)
        
        #apply lambda
        print('apply lambda')
        train[fixed_label] = lambdaDataset(train,fixed_label)
        
        #tokenize
        print('tokenize')
        tokenized_tweet =  tokenize(train,fixed_label)
        
        #tokenize porter
        print('tokenize porter')
        tokenized_tweet = tokenizePortStemmer(tokenized_tweet)
        
        #apply tokenize
        print('apply tokenize')
        train = applyTokenize(tokenized_tweet,train,fixed_label)
        print(train)
        
        #TF-IDF
        print('TF-IDF')
        
        tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=20000,stop_words='english')

        tfidf_matrix=tfidf.fit_transform(train[fixed_label])

        df_tfidf = pd.DataFrame(tfidf_matrix.todense())
        
        
        textCleaner(df = train,src='text',dst=fixed_label,stop_words = 'english')
        
        print(tfidf_matrix)
        train_tfidf_matrix = tfidf_matrix[:40000]

        #print(train_tfidf_matrix.todense())
        
        
        x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)
        fitModels(x_train_tfidf,y_train_tfidf,x_valid_tfidf,y_valid_tfidf)     
        
    else:
        print("already fitted")
        
    return log_acc,xgb_acc,dec_acc

    
    
  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()
    

    
    main()
    
    text = 'According to the company s updated strategy for the years 2009-2012  Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales .'
    
    log,xgb,dec = predict(text,'selam',fixed_label)
    
    print("after")
    print(log)
    print(xgb)
    print(dec)
    