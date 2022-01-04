import collections
import numpy
import Crawler
import SentAnalysis
import DatabaseService


def SentimentalAnalysis(url):
    SentAnalysis.TrainModels()

    text = Crawler.Crawl(url)

    log,xgb,dec = SentAnalysis.predict(text,"predict","fixed")

    xgb = xgb.tolist()
    dec = dec.tolist()


    if(DatabaseService.CheckUrlIsAlreadyExists(url)):
        DatabaseService.UpdateDocument(url, log, xgb, dec)
    else:
        DatabaseService.InsertToDatabase(url, log, xgb, dec)

    return{
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
                }
            
        }



