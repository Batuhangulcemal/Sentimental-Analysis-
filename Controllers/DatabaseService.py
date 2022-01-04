import collections
import pymongo
from bson.objectid import ObjectId
from datetime import datetime

client = pymongo.MongoClient("mongodb+srv://test:test@sentanalysisproject.1qgcc.mongodb.net/SentimentalAnalysis?retryWrites=true&w=majority")
database = client["SentimentalAnalysis"]
collection = database["Analysis"]



def InsertToDatabase(link, log, xgb, dec):

    insert = {
        "link": link,
        "log": 
            {
                "neutral" : log[0][0],
                "positive" : log[0][1],
                "negative" : log[0][2]
            },
        "xgb":
            {
                "neutral" : xgb[0][0],
                "positive" : xgb[0][1],   
                "negative" : xgb[0][2]                 
            },
        "dec":
            {
                "predict" : dec[0]
            },
        "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    collection.insert_one(insert)

def UpdateDocument(link, log, xgb, dec):

    insert = {
        "$set":
        {
            "log": 
                {
                    "neutral" : log[0][0],
                    "positive" : log[0][1],
                    "negative" : log[0][2]
                },
            "xgb":
                {
                    "neutral" : xgb[0][0],
                    "positive" : xgb[0][1],   
                    "negative" : xgb[0][2]                 
                },
            "dec":
                {
                    "predict" : dec[0]
                },
            "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
    }

    collection.update_one({"link": link}, insert)


def CheckUrlIsAlreadyExists(url):
    if collection.count_documents({"link": url}, limit = 1) != 0: #is document exist
        return True
    return False

