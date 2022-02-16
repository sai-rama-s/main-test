'''
Collocation
'''

# Imports
import rasterio
from pyproj import Proj, transform
from rasterio import plot
import matplotlib.pyplot as plt
import boto3
import rasterio
from rasterio import plot
import matplotlib.pyplot as plt
import boto3
import rasterio 
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
from rasterio.crs import CRS
import os
import numpy as np
import zipfile

from rasterio.warp import calculate_default_transform, reproject, Resampling

# Main Functions
# Coords Functions
def getFeatures(gdf):
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def convertCRS(inPoint):
    lat,lon = inPoint
    P32640 = Proj(init='epsg:32640')
    P4326 = Proj(init='epsg:4326')
    x,y = transform(P4326, P32640, lon, lat)
    return x,y

# Clip Functions
def clip_tile(band, rtype, coords):
    out_img, out_transform = mask(dataset=band, shapes=coords, crop=True)
    out_meta = band.meta.copy()
    epsg_code = int(band.crs.data['init'][5:])
    out_meta.update({"driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
    )
    out_tif = 'clipped_f' + rtype + '.tif'
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)

def getCoord(minx, miny, maxx, maxy):
    bbox = box(minx, miny, maxx, maxy)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=CRS.from_epsg(32640))
    geo = geo.to_crs(CRS.from_epsg(32640))
    coords = getFeatures(geo)
    return coords

# Normalisation Functions
def Normalise_MinMax(I):
    I = np.copy(I)
    I_max = np.max(I)
    I_min = np.min(I)
    I = (I - I_min) / (I_max - I_min)
    return I

def Normalise_HistogramNorm(I):
    I = np.copy(I)
    I[I == 0] = np.nan
    mean = np.nanmean(I)
    std =  2 * np.nanstd(I)
    I[I > mean + std] = mean + std
    I = I - (mean - std)
    I[I < 0] = np.nan
    I = I / np.nanmax(I)
    I[I != I] = 0
    return I

def Collocate(working_dir="./"):
    # Connect to S3
    s3_client = boto3.client('s3',
        aws_access_key_id='AKIAQVSA7EXYD5OVVQMD',
        aws_secret_access_key='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'
    )
    #s3_client.download_file('sentinel-api-test', 'S1A_IW_GRDH_1SDV_20211203T021544_20211203T021607_040838_04D934_0E4B.zip', 'S1A_IW_GRDH_1SDV_20211203T021544_20211203T021607_040838_04D934_0E4B.zip')
    #s3_client.download_file('sentinel-api-test', 'S2A_MSIL1C_20211209T065251_N0301_R020_T40RBN_20211209T074434.zip', 'S2A_MSIL1C_20211209T065251_N0301_R020_T40RBN_20211209T074434.zip')

    #  with zipfile.ZipFile("S1A_IW_GRDH_1SDV_20211203T021544_20211203T021607_040838_04D934_0E4B.zip","r") as zip_ref:
    #         zip_ref.extractall("./")
    #    with zipfile.ZipFile("S2A_MSIL1C_20211209T065251_N0301_R020_T40RBN_20211209T074434.zip","r") as zip_ref:
    #   zip_ref.extractall("./")

    # Download and Load Coords File
    SAVE_PATH_COORDS = working_dir + "test.txt"
    s3_client.download_file('sentinel-api-txt', 'demo-website/hello.txt', SAVE_PATH_COORDS)
    Text_File_Import = open(SAVE_PATH_COORDS, 'r')
    Text_lines = Text_File_Import.readlines()

    #print('1st',Text_lines[0])
    #print('2nd', Text_lines[1])
    l1, l2 = Text_lines[0].split(' ')
    r1, r2 = Text_lines[1].split(' ')
    l1 = float(l1)
    l2 = float(l2[:-1])
    print('1st', l1, l2)
    #r1, r2 = Text_lines[1].split(' ')
    r1 = float(r1)
    r2 = float(r2)
    print('2nd', r1, r2)

    # Convert CRS
    #l1, l2, r1, r2
    print(l1, l2, r1, r2)
    minx, miny = convertCRS((l1,l2))
    maxx,maxy = convertCRS((r1,r2))
    print(minx,miny,maxx,maxy)

    # Download and Load Raw Optical and SAR Bands
    SAVE_PATH_RESAMPLEDSAR = working_dir + "resampled_sar.tif"
    s3_client.download_file('sentinel-api-test', 'resampled_sar.tif', SAVE_PATH_RESAMPLEDSAR)
    OPTICAL_IMAGE_PATH = "./S2A_MSIL1C_20211209T065251_N0301_R020_T40RBN_20211209T074434.SAFE/GRANULE/L1C_T40RBN_A033765_20211209T065253/IMG_DATA/"
    b = rasterio.open(OPTICAL_IMAGE_PATH + 'T40RBN_20211209T065251_B02.jp2', driver='JP2OpenJPEG')
    g = rasterio.open(OPTICAL_IMAGE_PATH + 'T40RBN_20211209T065251_B03.jp2', driver='JP2OpenJPEG')
    r = rasterio.open(OPTICAL_IMAGE_PATH + 'T40RBN_20211209T065251_B04.jp2', driver='JP2OpenJPEG')
    
    #SAR_IMAGE_PATH = "./S1A_IW_GRDH_1SDV_20211203T021544_20211203T021607_040838_04D934_0E4B.SAFE/measurement/"
    sar = rasterio.open(SAVE_PATH_RESAMPLEDSAR)
    #imagePath1 = './S1A_IW_GRDH_1SDV_20211203T021544_20211203T021607_040838_04D934_0E4B.SAFE/measurement/'
    #band = rasterio.open(imagePath1+'s1a-iw-grd-vv-20211203t021544-20211203t021607-040838-04d934-001.tiff'
    #band = rasterio.open(imagePath1+'s1a-iw-grd-vv-20211203t021544-20211203t021607-040838-04d934-001.tiff'
    #minx, miny, maxx, maxy = 254880.0,2800020.0,309780.0,2690220.0

    # Get Coords
    coords1 = getCoord(minx,miny, maxx, maxy)
    print(coords1)

    # Clip Band Tiles
    clip_tile(r, 'optical_r',coords1)
    clip_tile(g, 'optical_g',coords1)
    clip_tile(b, 'optical_b', coords1)
    clip_tile(sar, 'sar', coords1)
    clipped_sar = rasterio.open('clipped_fsar.tif').read()
    clipped_r =  rasterio.open('clipped_foptical_r.tif').read()
    clipped_g =  rasterio.open('clipped_foptical_g.tif').read()
    clipped_b =  rasterio.open('clipped_foptical_b.tif').read()
    # if clipped_sar.shape != clipped_r.shape:
    #     clipped_r = clipped_r[:, :clipped_sar.shape[1], :clipped_sar.shape[2]]
    #     clipped_g = clipped_g[:, :clipped_sar.shape[1], :clipped_sar.shape[2]]
    #     clipped_b = clipped_b[:, :clipped_sar.shape[1], :clipped_sar.shape[2]]
    # print(clipped_r.shape, clipped_g.shape, clipped_b.shape, clipped_sar.shape)
    rgb_list = [clipped_r[0,:,:], clipped_g[0,:,:], clipped_b[0,:,:]]

    # Normalise and Save Optical
    rgb_norm_list = []
    for band in rgb_list:
        band_norm = Normalise_MinMax(band)
        band_norm = Normalise_HistogramNorm(band_norm)
        rgb_norm_list.append(band_norm)
    I_rgb_norm = np.dstack(tuple(rgb_norm_list))
    plt.figure(figsize=(15, 40), dpi = 10000)
    plt.axis('off')
    #plt.imshow(I_rgb_norm)
    #plt.savefig('Optical.jpg', format="jpeg", bbox_inches='tight')
    #plt.show()
    SAVE_PATH_FINAL_OPTICAL = working_dir + "Optical.jpg"
    plt.imsave(SAVE_PATH_FINAL_OPTICAL, I_rgb_norm, dpi = 10000)

    # Normalise and Save SAR
    sar_clipped = clipped_sar[0,:,:]
    print(np.max(sar_clipped))
    band_norm = Normalise_MinMax(sar_clipped)
    band_norm = Normalise_HistogramNorm(band_norm)
    plt.figure(figsize=(15, 40), dpi = 10000)
    #plt.imshow(clipped_sar[0,:,:], cmap = 'gray', vmax = 500 , vmin = 20)
    plt.axis('off')
    #plt.savefig(f"Sar.jpg", format="jpeg",  bbox_inches='tight')
    #plt.show()
    SAVE_PATH_FINAL_SAR = working_dir + "Sar.jpg"
    plt.imsave(SAVE_PATH_FINAL_SAR,band_norm,dpi = 10000, cmap = 'gray')

    # Upload to S3
    s3_client.upload_file(SAVE_PATH_FINAL_OPTICAL, 'sentinel-api-test', 'demo-website/Optical.jpg')
    s3_client.upload_file(SAVE_PATH_FINAL_SAR, 'sentinel-api-test', 'demo-website/Sar.jpg')

# Driver Code
if __name__ == "__main__":
    Collocate()