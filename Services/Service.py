import collections
import pymongo

client = pymongo.MongoClient("mongodb+srv://test:test@sentanalysisproject.1qgcc.mongodb.net/testDb?retryWrites=true&w=majority")
database = client["testDb"]
collection = database["testCol"]

collection.insert_one(
    {"name": "batu",
    "surname": "gulcemal"})




