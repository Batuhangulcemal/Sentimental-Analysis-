import numpy as np
import pandas as pd
from flask import Flask, request
from flask_restful import Resource, Api
from flask_pymongo import PyMongo


client = PyMongo.MongoClient()

app = Flask(__name__)
api = Api(app)

List = list()

class Hello(Resource):
    def get(self,str):
        List.append(str)
        return {"Hello": str}

class Print(Resource):
    def get(self):
        print(type({"Hello": str}))
        return 



api.add_resource(Hello, "/<string:str>")
api.add_resource(Print, "/print") 

    

app.run(debug=True)

