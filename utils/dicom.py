import copy

import pydicom
import pydicom.pixel_data_handlers
from pydicom.encaps import encapsulate

is_compresssed = lambda dataset: dataset.file_meta.TransferSyntaxUID not in pydicom.uid.UncompressedTransferSyntaxes


def read_dicom(file_name):
    ds = pydicom.dcmread(file_name)
    if is_compresssed(ds):
        ds.decompress()
    ds.SamplesPerPixel = 1 
    ds.PhotometricInterpretation = 'MONOCHROME2'
    return ds

def write_dicom(new_array, dicom_data):        
    new_data = copy.deepcopy(dicom_data)

    if new_array.ndim == 2: 
        new_data.NumberOfFrames = 1
        new_data.Rows, new_data.Columns = new_array.shape
    elif new_array.ndim == 3:
        new_data.NumberOfFrames, new_data.Rows, new_data.Columns = new_array.shape

    if is_compresssed(dicom_data):
        new_data.PixelData = encapsulate(new_array.tobytes())
    else:
        new_data.PixelData = new_array.tobytes()
        
    return new_data