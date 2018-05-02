#!/usr/bin/python3
# -*- coding:utf-8 -*-
import base64
import json
import multiprocessing
import os
import traceback
from urllib import parse
import sys
import codecs
import requests
import time

from flask import Flask, request
from flask_cors import CORS
from flask_script import Manager


import numpy as np
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def create_app():
    app = Flask(__name__)
    CORS(app)

    process_queue = {}

    @app.route('/', methods=['GET', 'POST'])
    def home():
        return 'hello world!'

    @app.route('/training/<path:job>', methods=['GET', 'POST'])
    def training(job):
        def action():
            pass



    # run the job in separate process not thread
    @app.route('/run_trainingjob/<path:jobx>', methods=['GET'])
    def run_trainingjob(jobx):
        print(time.time())
        gpu_id = request.args.get('gpu_id')
        def action():
            job = jobservice.getJobByPath(job_path=jobx)
            if (len(job) > 0):
                jobObj = job[0]
                ui_train = UI_Train(plugin_path=jobObj['algchain_path'],
                                    dataset_path=jobObj['dataset_path'],
                                    user_path=os.path.join(user_path, jobObj['user_path']),
                                    gpu_id=gpu_id,
                                    split_ratio=jobObj['split_ratio'])
                ui_train.train_run(jobx_path=parse.unquote(jobx))

        try:
            print(time.time())
            t = multiprocessing.Process(target=action)
            t.start()
            print(time.time())

            process_queue[jobx] = t

            msg = "Succ"
            state = 1

        except Exception as e:
            print(e)
            msg = "Error: " + str(e)
            state= 0
        dic = {'state':state,'msg':msg}
        print(time.time())
        return json.dumps(dic,cls=MyEncoder)


manager = Manager(create_app)

if __name__ == "__main__":
    manager.run()
