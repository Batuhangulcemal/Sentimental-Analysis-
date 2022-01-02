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
        log,xgb,dec = Service.Gotten((args["task"]))
        xgb  = xgb.tolist()
        dec = dec.tolist()
        print("selammmm")
        print(log)
        print(xgb)
        print(dec)
        return{
            "log": 
                {
                    "1" : log[0][0],
                    "2" : log[0][1],
                    "3" : log[0][2]
                },
            "xgb":
                {
                    "1" : float(xgb[0][0]),
                    "2" : float(xgb[0][1]),   
                    "3" : float(xgb[0][2])                 
                },
            "dec":
                {
                    "1" : dec[0]
                }
            
        }


api.add_resource(Hello, "/<string:str>")
api.add_resource(Print, "/print") 
api.add_resource(Gotten,"/gotten")

    

app.run()

