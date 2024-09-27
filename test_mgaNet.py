__AUTHOR__ = 'Bahram Jafrasteh'

"""
This script accepts a file with format xx.nii.gz and generate a mask with format xx_mask.nii.gz and a reconstructed file with format xx_rec.nii.gz
please be careful in selection of eco_mri variable that should be 1 for MRI and -1 for US images.
you can change the threshold value as you want to improve the brain extraction.
"""
from model.mga_net import MGA_NET

import torch
import sys
import os
from model.utils import *
from scipy.ndimage import binary_fill_holes

file_inp = sys.argv[1]
eco_mri = int(sys.argv[2]) # -1 for US and 1 for MRI
threshold = 0.0
if len(sys.argv)>3:
    threshold = float(sys.argv[3])

high_quality_rec = True # Network has been trained on 128x128x128 size image. However, it is possible to sample 192x192x192 images to get higher quality images
basen = os.path.basename(file_inp)
basen = basen[:basen.find('.nii')]

file_inp_mask = os.path.join(os.path.dirname(file_inp),basen+'_mask.nii.gz')
file_inp_rec = os.path.join(os.path.dirname(file_inp),basen+'_rec.nii.gz')
if torch.cuda.is_available():
    # device = torch.device("cuda")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
#else:
device = torch.device("cpu")


################## Loading model ################################
model = MGA_NET(time_embed=True)
model.to(device)
load_filepath = 'MGA_NET.pth'
state_dict = torch.load(load_filepath, map_location=device)
model.load_state_dict(state_dict['model'], strict=True)



################## Loading data and data standardization ################################
time = torch.from_numpy(np.array(eco_mri)).unsqueeze(0).to(torch.float).to(device)
imA = nib.load(file_inp)
affine_initial = imA.affine
imA_initial = imA.get_fdata().copy()
shape_initial = imA.shape
header_initial = imA.header

transform, source = convert_to_ras(imA.affine, target='RAS')
if source != 'RAS':
    imA = imA.as_reoriented(transform)

pixdim = imA.header['pixdim'][1:4]
affine = imA.affine
header = imA.header
imA = imA.get_fdata()
border_value = imA[0, 0, 0]
image_used, pad_zero = remove_zero(imA, border_value)

image_used = normalize_mri(image_used)/255.0
shape_zero = image_used.shape

target_shape = [128, 128, 128]
image_used_1 = resample_to_size(nib.Nifti1Image(image_used, affine, header), new_size=target_shape,
                                   method='spline').get_fdata()

################## Brain extraction and image reconstruction ################################
imB = torch.from_numpy(image_used_1).to(torch.float).unsqueeze(0).unsqueeze(0)
imB = imB.to(device)
im_low = model.forward(imB, time)
im_mask_low, im_rec_low = im_low

if high_quality_rec:


    target_shape = [192, 192, 192] # to create higher quality images
    image_used_2 = resample_to_size(nib.Nifti1Image(image_used, affine, header), new_size=target_shape,
                              method='spline').get_fdata()

    imA = torch.from_numpy(image_used_2).to(torch.float).unsqueeze(0).unsqueeze(0)
    imA = imA.to(device)
    im_high = model.forward(imA, time)

    im_mask_high, im_rec_high = im_high

    im_rec = im_rec_high.detach().cpu().squeeze().numpy()


    im_mask = im_mask_low.detach().cpu().squeeze().numpy()
    im_rec = im_rec_high.detach().cpu().squeeze().numpy()
else:
    im_mask = im_mask_low.detach().cpu().squeeze().numpy()
    im_rec = im_rec_low.detach().cpu().squeeze().numpy()

################## resmaple to the original size ################################
im_mask = resample_to_size(nib.Nifti1Image(im_mask, affine, header), new_size=shape_zero,
                           method='spline').get_fdata()
im_rec = resample_to_size(nib.Nifti1Image(im_rec, affine, header), new_size=shape_zero,
                          method='spline').get_fdata()

im_mask = get_back_data(im_mask, shape_initial, pad_zero, im_mask[0, 0, 0])
im_rec = get_back_data(im_rec, shape_initial, pad_zero, im_rec[0, 0, 0])

header['pixdim'][1:4] = pixdim

mask = nib.Nifti1Image(im_mask, affine, header)


_, source = convert_to_ras(affine_initial)
transform, _ = convert_to_ras(mask.affine, target=source)
mask = mask.as_reoriented(transform)

im_rec = nib.Nifti1Image(im_rec, affine, header)
im_rec = im_rec.as_reoriented(transform)



################## mask preprocessing ################################

im_mask = mask.get_fdata().copy()

ind = im_mask >= threshold
im_mask[ind] = 0
im_mask[~ind] = 1

im_mask = binary_fill_holes(im_mask)
im_mask, labels_freq = LargestCC(im_mask, connectivity=1)
argmax = np.argmax(
    [imA_initial[im_mask == el].sum() for el in range(len(labels_freq)) if el != 0]) + 1

ind = im_mask != argmax
im_mask[ind] = 0
im_mask[~ind] = 1



a1 = normalize_mri(imA_initial * im_mask)
rec = normalize_mri(im_rec.get_fdata() * im_mask)

imB = im_rec.get_fdata().copy()
imB[~(im_mask > 0)] = 0
imB = normalize_mri(imB)

################## saving the results ################################
nib.Nifti1Image(imB, affine_initial, header_initial).to_filename(file_inp_rec)
nib.Nifti1Image(im_mask, affine_initial, header_initial).to_filename(file_inp_mask)
