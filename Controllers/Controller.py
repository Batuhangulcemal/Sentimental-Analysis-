from typing import Dict
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_restful import Resource, Api, reqparse
import Service


parser = reqparse.RequestParser()
parser.add_argument('task')

app = Flask(__name__)
api = Api(app)


class Url(Resource):
    def post(self):
        args = parser.parse_args()
        task = {'task': args['task']}
        return Service.SentimentalAnalysis((args["task"]))

@app.after_request

def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
  return response      


api.add_resource(Url,"/url")
app.run()

