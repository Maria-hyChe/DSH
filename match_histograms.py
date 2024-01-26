from skimage.exposure import match_histograms

from utils.utils import read_tif, writeTiff


reference_path='H:/America_s2/Landsat/s2201908_B.tif'
image_path='H:/America_s2/Landsat/L8_201908_B.tif'

_, _, reference = read_tif(reference_path)
im_proj, im_Geotrans, image = read_tif(image_path)

channel_axis = None

if len(reference.shape) > 2:
    print('multi channel!')
    reference = reference.swapaxes(1, 0)
    reference = reference.swapaxes(1, 2)
    print("reference shape :",reference.shape)
    image = image.swapaxes(1, 0)
    image = image.swapaxes(1, 2)
    print("image shape :",image.shape)
    channel_axis = 2

print("channel_axis:",channel_axis)
matched =match_histograms(image,reference,channel_axis=channel_axis)
print("matched image shape :",matched.shape)

if len(reference.shape) > 2:
    matched = matched.swapaxes(1, 2)
    matched = matched.swapaxes(0, 1)
print("matched shape :",matched.shape)

writeTiff("H:\match_result\L8_201908_B_mat_s2.tif", matched, im_Geotrans, im_proj)
print("finish!")

