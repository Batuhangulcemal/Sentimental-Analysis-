import collections
import pymongo
from datetime import datetime

client = pymongo.MongoClient("mongodb+srv://test:test@sentanalysisproject.1qgcc.mongodb.net/SentimentalAnalysis?retryWrites=true&w=majority")
database = client["SentimentalAnalysis"]
collection = database["Analysis"]

print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

def InsertToDatabase(link, log, xgb, dec):

    insert = {
        "link": link,
        "log": 
            {
                "1" : log[0][0],
                "2" : log[0][1],
                "3" : log[0][2]
            },
        "xgb":
            {
                "1" : xgb[0][0],
                "2" : xgb[0][1],   
                "3" : xgb[0][2]                 
            },
        "dec":
            {
                "1" : dec[0]
            },
        "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    collection.insert_one(insert)
