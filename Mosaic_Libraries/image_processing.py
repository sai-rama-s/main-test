import rasterio
import pyproj

from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon
from rasterio.mask import mask
from rasterio.merge import merge
from Mosaic_Libraries.global_params import *


def coordsys(image_filepaths, text_fp):
    try:
        with rasterio.open(image_filepaths[0]) as src:
            f = open(text_fp, "w+")
            f.write(str(src.meta['crs']))
            f.close()
            return src.meta['crs']

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


def coord_transform(intersections, crsystem):
    try:
        transformer = pyproj.Transformer.from_crs("epsg:4326", crsystem)
        coords = []
        for image in intersections:
            local_coords = []
            for crs in image[:-1]:
                local_coords.append(transformer.transform(crs[1], crs[0]))

            coords.append(local_coords)

        return coords
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


def sar_reproj(img_path, dest_path, crsystem):
    try:
        with rasterio.open(img_path) as src:
            gcps, gcp_crs = src.gcps
            transform, width, height = calculate_default_transform(gcp_crs, crsystem, src.width, src.height, gcps=gcps)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': crsystem,
                'transform': transform,
                'width': width,
                'height': height
            })

            with rasterio.open(dest_path, 'w', **kwargs) as dst:
                for j in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, j),
                        destination=rasterio.band(dst, j),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crsystem,
                        resampling=Resampling.nearest)
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


def poly_list(aoi_coords):
    try:
        polygon = []
        for i in range(len(aoi_coords)):
            polygon.append(Polygon(aoi_coords[i]))

        return polygon
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


def aoi_crop(fps, polygons):
    try:
        crop_img = []
        crop_trans = []
        for i in range(len(fps)):
            f = rasterio.open(fps[i])
            out, out_trans = mask(f, [polygons[i]], crop=True)
            crop_img.append(out)
            crop_trans.append(out_trans)

        return crop_img, crop_trans
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


def stitch(img_fps):
    try:
        src_files_to_mosaic = []

        for sim in img_fps:
            src = rasterio.open(sim)
            src_files_to_mosaic.append(src)

        return merge(src_files_to_mosaic)
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

def save_image(image, outpath, ref_fp, crs, transform, img_type="Gtiff", bands=1):
    try:
        input_dtype = rasterio.open(ref_fp).dtypes[0]
        with rasterio.open(outpath, "w", driver=img_type, count=bands,
                           height=image.shape[1],
                           width=image.shape[2],
                           transform=transform,
                           crs=crs,
                           dtype=input_dtype) as dest:
            dest.write(image)
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