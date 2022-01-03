import collections
import json
from json import JSONEncoder
import numpy
import Crawler
import SentAnalysis
import DatabaseService


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)    

def SentimentalAnalysis(url):
    SentAnalysis.TrainModels()

    text = Crawler.Crawl(url)

    log,xgb,dec = SentAnalysis.predict(text,"predict","fixed")

    xgb = xgb.tolist()
    dec = dec.tolist()

    DatabaseService.InsertToDatabase(url, log, xgb, dec)

    return{
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
                }
            
        }



