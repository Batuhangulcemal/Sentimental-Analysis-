from typing import Dict
import numpy as np
import pandas as pd
import Link
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_pymongo import PyMongo
import Service


parser = reqparse.RequestParser()
parser.add_argument('task')

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

class Gotten(Resource):
    def post(self):
        args = parser.parse_args()
        task = {'task': args['task']}
        print(args["task"])
        #return {Service.Gotten((args["task"]))[0]}
        return{
            "log": Service.Gotten((args["task"]))[0],
            "xgb": Service.Gotten((args["task"]))[1],
            "dec": Service.Gotten((args["task"]))[2]
        }


api.add_resource(Hello, "/<string:str>")
api.add_resource(Print, "/print") 
api.add_resource(Gotten,"/gotten")

    

app.run()

