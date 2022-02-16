from Mosaic_Libraries.global_params import *
import glob
import os
import json


# Defining all functions used


def file_search(search_criteria, search_path, current_only=False):
    result = []
    try:
        if not current_only:
            for root, dirs, files in os.walk(search_path, ):
                q = os.path.join(root, search_criteria)
                img_fps = glob.glob(q)

                for i in img_fps:
                    result.append(i)

            if not result:
                raise NotImplementedError(
                    "Error: File path search failed. The required files were not found in this directory.")
            else:
                return result

        else:
            q = os.path.join(search_path, search_criteria)
            result = glob.glob(q)
            if not result:
                raise NotImplementedError(
                    "Error: File path search failed. The required files were not found in this directory.")
            else:
                return result
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


def data_json(filepath, data_type):
    try:
        f = open(filepath)
        data = json.load(f)
        intersections = []
        titles = []
        coverWholeAoi = False
        if len(data[data_type]) == 1:
            coverWholeAoi = True
            tile = data[data_type][0]
            titles.append(tile['title'])
            intersections.append(tile['intersection'][0])
        else:
            for tile in data[data_type]:
                titles.append(tile['title'])
                intersections.append(tile['intersection'][0])

        f.close()

        return titles, intersections, coverWholeAoi
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


def get_fps(search_path, titles, search_criteria):
    ordered_fps = []
    given_fps = file_search(search_criteria, search_path)

    for title in titles:
        for fp in given_fps:
            if title in fp:
                ordered_fps.append(fp)
                given_fps.remove(fp)
                break

    return ordered_fps
