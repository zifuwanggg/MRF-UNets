import os
import sys
from glob import glob

import numpy as np
from PIL import Image
from osgeo import gdal, ogr # only required for DeepGlobe Building
import pydicom # only required for CHAOS
import SimpleITK as sitk # only required for PROMISE

from datas.data_utils import expand_image_list, get_label_file


# https://gist.github.com/avanetten/b295e89f6fa9654c9e9e480bdb2e4d60#file-create_building_mask-py
def create_poly_mask(rasterSrc, vectorSrc, npDistFileName='', noDataValue=0, burn_values=1):
    '''
    Create polygon mask for rasterSrc,
    Similar to labeltools/createNPPixArray() in spacenet utilities
    '''
    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    if npDistFileName == '':
        dstPath = ".tmp.tiff"
    else:
        dstPath = npDistFileName

    ## create First raster memory layer, units are pixels
    ## change output to geotiff instead of memory 
    memdrv = gdal.GetDriverByName('GTiff') 
    dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte, 
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)    
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0

    if npDistFileName == '':
        os.remove(dstPath)
        
        
# https://stackoverflow.com/questions/9744255/instagram-lux-effect/9761841#9761841
def auto_contrast(image):
    hist = histogram(image)
    p5 = shade_at_percentile(hist, .01)
    p95 = shade_at_percentile(hist, .99)
    a = 255.0 / (p95 + p5)
    b = -1.0 * a * p5

    result = (image.astype(float) * a) + b
    result = result.clip(0, 255.0)
    
    return result


def histogram(image):
    hist = dict()
    for shade in range(0, 256):
        hist[shade] = 0
    for _, val in np.ndenumerate(image):
        hist[val] += 1
    
    return hist


def shade_at_percentile(hist, percentile):
    n = sum(hist.values())
    cumulative_sum = 0.0
    for shade in range(0, 256):
        cumulative_sum += hist[shade]
        if cumulative_sum / n >= percentile:
            return shade
    
    return None


def Land(data_dir):
    DeepGlobe(data_dir, 'land')
    
    
def Road(data_dir):
    DeepGlobe(data_dir, 'road')
    
    
def Building(data_dir):
    print(f"Processing SpaceNet")
    src_dir = f'{data_dir}/building/spacenet'
    dst_dir = f'{data_dir}/building/train'
    
    AOIs = {'Vegas': 2, 'Paris': 3, 'Shanghai': 4, 'Khartoum': 5}
    options_list = ['-ot Byte',
                    '-of PNG',
                    '-b 1',
                    '-b 2',
                    '-b 3',
                    '-scale']
    options_string = " ".join(options_list)
        
    for city, AOI in AOIs.items():
        raster_dir = f'{src_dir}/AOI_{AOI}_{city}_Train/RGB-PanSharpen/*'
        vector_dir = f'{src_dir}/AOI_{AOI}_{city}_Train/geojson/buildings/'
        for raster_file in glob(raster_dir):
            tif = raster_file.split('/')[-1].split('_')[-1]
            index = int(''.join((filter(str.isdigit, tif))))

            image_file = f'{city}_{index}_sat.jpg'
            label_file = f'{city}_{index}_mask.png'
            vector_file = f'buildings_AOI_{AOI}_{city}_img{index}.geojson'
            
            image_file = os.path.join(dst_dir, image_file)
            label_file = os.path.join(dst_dir, label_file)
            vector_file = os.path.join(vector_dir, vector_file)

            gdal.Translate(image_file, raster_file, options=options_string)
            create_poly_mask(raster_file, vector_file, label_file, noDataValue=0, burn_values=255)
            
    DeepGlobe(data_dir, 'building')
        
    
def DeepGlobe(data_dir, dataset):
    print(f"Processing DeepGlobe {dataset}")
    src_dir = os.path.join(data_dir, f'{dataset}/train')
    dst_dir = os.path.join(data_dir, f'{dataset}/resized')
    os.makedirs(dst_dir, exist_ok=True)

    image_path = os.path.join(src_dir, '*_sat.jpg')
    label_path = os.path.join(src_dir, '*_mask.png')
    
    image_list = glob(image_path)
    label_list = glob(label_path)

    for image_file in image_list:
        image_resized_file = image_file.split('/')[-1].replace('.jpg', '.png')
        image_resized_file = os.path.join(dst_dir, image_resized_file)
        
        image = Image.open(image_file)
        image_resized = image.resize((256, 256), resample=Image.BILINEAR)
        image_resized.save(image_resized_file)

    for label_file in label_list:
        label_resized_file = label_file.split('/')[-1]
        label_resized_file = os.path.join(dst_dir, label_resized_file)
        
        label = Image.open(label_file)
        label_resized = label.resize((256, 256), resample=Image.NEAREST)
        label_resized.save(label_resized_file)
        
        
def CHAOS(data_dir):
    print("Processing CHAOS")
    src_dir = os.path.join(data_dir, 'chaos/train/MR/*')

    image_list = glob(src_dir)
    image_list = expand_image_list(image_list, dataset='chaos')

    for image_file in image_list:
        image_dst_dir = image_file.replace('train/MR', 'resized')[:-19]
        label_dst_dir = image_dst_dir.replace('DICOM_anon', 'Ground')
        if 'InPhase' in image_dst_dir:
            label_dst_dir = label_dst_dir.replace('/InPhase', '')
        elif 'OutPhase' in image_dst_dir:
            label_dst_dir = label_dst_dir.replace('/OutPhase', '')
        os.makedirs(image_dst_dir, exist_ok=True)
        os.makedirs(label_dst_dir, exist_ok=True)
        
        dicom = pydicom.dcmread(image_file)
        image = dicom.pixel_array.astype(float)
        image_scaled = 255 * (np.maximum(image, 0) / image.max())
        image_scaled = np.uint8(image_scaled)
        image_contrast = auto_contrast(image_scaled)
        image_contrast = np.uint8(image_contrast)
        image_array = Image.fromarray(image_contrast)
        image_resized = image_array.resize((256, 256), resample=Image.BILINEAR)
        
        image_file = image_file.replace('.dcm', '.png')
        label_file = get_label_file(image_file, dataset='chaos')
        
        label = Image.open(label_file)
        label_resized = label.resize((256, 256), resample=Image.NEAREST)

        image_resized_file = image_file.split('/')[-1]
        image_resized_file = os.path.join(image_dst_dir, image_resized_file)
        label_resized_file = label_file.split('/')[-1]
        label_resized_file = os.path.join(label_dst_dir, label_resized_file)
            
        image_resized.save(image_resized_file)
        if not os.path.isfile(label_resized_file): 
            label_resized.save(label_resized_file)


def PROMISE(data_dir):
    print("Processing PROMISE")
    src_dir_1 = os.path.join(data_dir, 'promise/train/TrainingData_Part1/*')
    src_dir_2 = os.path.join(data_dir, 'promise/train/TrainingData_Part2/*')
    src_dir_3 = os.path.join(data_dir, 'promise/train/TrainingData_Part3/*')
    dst_dir = os.path.join(data_dir, 'promise/resized')
    os.makedirs(dst_dir, exist_ok=True)

    image_list = glob(src_dir_1) + glob(src_dir_2) + glob(src_dir_3)
    image_list = sorted((f for f in image_list if '.mhd' in f and 'segmentation' not in f))

    for image_file in image_list:
        case = int(image_file[-6:-4])
        case_dir = os.path.join(dst_dir, str(case))
        os.mkdir(case_dir)
    
        image_mhd = sitk.ReadImage(image_file)
        label_mhd = sitk.ReadImage(image_file[:-4] + '_segmentation.mhd')

        image = sitk.GetArrayFromImage(image_mhd).astype(float)
        label = sitk.GetArrayFromImage(label_mhd).astype(float)

        for i in range(image.shape[0]):
            image_scaled = 255 * (np.maximum(image[i], 0) / image[i].max())
            image_scaled = np.uint8(image_scaled)
            image_contrast = auto_contrast(image_scaled)
            image_contrast = np.uint8(image_contrast)
            image_array = Image.fromarray(image_contrast)
            image_resized = image_array.resize((256, 256), resample=Image.BILINEAR)
        
            label_scaled = 255 * np.maximum(label[i], 0)
            label_scaled = np.uint8(label_scaled)
            label_array = Image.fromarray(label_scaled)
            label_resized = label_array.resize((256, 256), resample=Image.NEAREST)

            image_resized_file = os.path.join(case_dir, f'{i}_image.png')
            label_resized_file = os.path.join(case_dir, f'{i}_mask.png')
            
            image_resized.save(image_resized_file)
            label_resized.save(label_resized_file)

                    
if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2])