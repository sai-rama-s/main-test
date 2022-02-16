from sentinelsat import SentinelAPI
from Mosaic_Libraries.global_params import *


import json
import zipfile


# Defining all functions used


def ids_json(filepath):
    try:
        f = open(filepath)
        data = json.load(f)

        op_ids = []
        sar_ids = []

        for i in data['optical']:
            op_ids.append(i['id'])

        for i in data['sar']:
            sar_ids.append(i['id'])

        f.close()
        print("Product ID's read from JSON file successfully. \n")
        return op_ids, sar_ids
    except:
        tmp = query_handler(order_id, "Error")
        instance_id = tmp['instance_id']
        s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                              PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
        ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
        ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")


def download_data(user_id, pwd, ids, filepath):
    try:
        api = SentinelAPI(user_id, pwd, 'https://scihub.copernicus.eu/dhus')

        for p_id in ids:
            api.download(p_id, filepath)

    except:
        tmp = query_handler(order_id, "Error")
        instance_id = tmp['instance_id']
        s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                              PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
        ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
        ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")


def unzip_files(filepaths):
    try:
        for filepath in filepaths:
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(filepath[:-4])

    except:
        tmp = query_handler(order_id, "Error")
        instance_id = tmp['instance_id']
        s3_client.upload_file("./log.txt", PARAMS["outputs"]["s3_bucket"],
                              PARAMS["outputs"]["s3_path"] + order_id + '/' + 'log.txt')
        ecs = boto3.client('ecs',
                               aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
                               aws_secret_access_key='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',
                               region_name='ap-south-1')
        ecs.stop_task(task=instance_id, cluster="airborne-cluster", reason="Terminated due to error")