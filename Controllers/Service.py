import collections
import json
from json import JSONEncoder
import numpy
import pymongo
import Crawler
import SentAnalysis


client = pymongo.MongoClient("mongodb+srv://test:test@sentanalysisproject.1qgcc.mongodb.net/testDb?retryWrites=true&w=majority")
database = client["testDb"]
collection = database["testCol"]

collection.insert_one(
    {"name": "batu",
    "surname": "gulcemal"})

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)    

def Gotten(url):
    SentAnalysis.TrainModels()

    text = Crawler.gotten(url)

    log,xgb,dec = SentAnalysis.predict(text,"predict","fixed")

    numpyData = {"array": log}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  # use dump() to write array into file
    numpyData1 = {"array": xgb}
    encodedNumpyData1 = json.dumps(numpyData1, cls=NumpyArrayEncoder)  # use dump() to write array into file
    numpyData2 = {"array": dec}
    encodedNumpyData2 = json.dumps(numpyData2, cls=NumpyArrayEncoder)  # use dump() to write array into file


    return encodedNumpyData, encodedNumpyData1, encodedNumpyData2



