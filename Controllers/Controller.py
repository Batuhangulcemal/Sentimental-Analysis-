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
                    "neutral" : log[0][0],
                    "positive" : log[0][1],
                    "negative" : log[0][2]
                },
            "xgb":
                {
                    "neutral" : float(xgb[0][0]),
                    "positive" : float(xgb[0][1]),   
                    "negative" : float(xgb[0][2])                 
                },
            "dec":
                {
                    "predict" : dec[0]
                }
            
        }
@app.after_request

def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response      

api.add_resource(Hello, "/<string:str>")
api.add_resource(Print, "/print") 
api.add_resource(Gotten,"/gotten")

    

app.run()

