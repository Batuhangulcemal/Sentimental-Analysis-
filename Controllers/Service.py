import collections
import numpy
from xgboost.callback import print_evaluation
import Crawler
import SentAnalysis
import DatabaseService
from flask import Flask, request, Response


def SentimentalAnalysis(url):
    
    text = Crawler.Crawl(url)
    if len(text) == 0:
        return Response(status=404)
    
    log_acc,xgb_acc,dec_acc = SentAnalysis.TrainModels()

    
    
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
                    "negative" : log[0][2],
                    "score" : {
                        "accuracy" :log_acc
                        }
                },
            "xgb":
                {
                    "neutral" : xgb[0][2],
                    "positive" : xgb[0][1],   
                    "negative" : xgb[0][0],
                    "score" : {
                        "accuracy" :xgb_acc
                        }               
                },
            "dec":
                {
                    "predict" : dec[0],
                    "score" : {
                        "accuracy" :dec_acc
                        }
                }
    }



