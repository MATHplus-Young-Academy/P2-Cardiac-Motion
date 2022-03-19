import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from skimage.io import imread, imshow
from skimage.measure import find_contours
from pydicom import dcmread
import nibabel as nib
import pyvista as pv
from IPython.display import HTML
import joblib
import pycpd

def cycle_plot(imgs, delay=400, axis=0, marker=None):
    if axis != 0:
        imgs = np.moveaxis(imgs, axis, 0)

    fig, ax = plt.subplots()

    frames = []
    for i, img in enumerate(imgs):
        content = []

        frame = ax.imshow(img, animated=True, cmap="Greys_r")
        content.append(frame)

        title = ax.text(
            0.5,
            1.02,
            str(i),
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize="large",
        )

        content.append(title)

        if marker:
            m = ax.scatter(*marker, c="red")
            content.append(frame)

        frames.append(content)

    plt.close()
    ani = animation.ArtistAnimation(fig, frames, interval=delay, blit=True)
    return HTML(ani.to_html5_video())

def load_labels(patient, basepath="/mnt/materials/SIRF/MathPlusBerlin/DATA/ACDC-Daten/DCM"):
    label_path = os.path.join(basepath, patient, "label.nii.gz")
    return nib.load(label_path).get_fdata()

def extract_contours(label):
    n_slices = label.shape[2]
    n_times = label.shape[4]

    contours = []
    for i in range(n_times):
        cnts_time = []
        for j in range(n_slices):
            cnts_slice = find_contours(label[:, :, j, 0, i, 0] == 2)
            if len(cnts_slice) == 1:
                cnts_time.append(cnts_slice[0])
            elif len(cnts_slice) > 1:
                raise RuntimeError("")
        contours.append(cnts_time)
    
    return contours

def segmentation_list_to_contours(segmentation_list):
    contours_all = []
    for i in range(len(segmentation_list)):
        contours_patient = []
        for j in range(segmentation_list[i].shape[0]):
            contours = []
            for k in range(segmentation_list[i].shape[1]):
                contours.append(find_contours(np.squeeze(segmentation_list[i][j,k,:,:])))
            contours_patient.append(contours)
        contours_all.append(contours_patient)
    return contours_all


def _callback(iteration, error, X, Y):
        print('iteration={0}, error={1}'.format(iteration, error))

def registration_affine(reference_contour, sample_contour, show_callback=True):
    
    reference_standardized = (reference_contour - reference_contour.mean(axis=0)) / reference_contour.std(axis=0)
    sample_standardized = (sample_contour - sample_contour.mean(axis=0)) / sample_contour.std(axis=0)
    reg = pycpd.affine_registration(X=reference_standardized, Y=sample_standardized)
    
    if show_callback:
        sample_aff, _ = reg.register(_callback)
    else:
        sample_aff, _ = reg.register()
        
    return reference_standardized, sample_standardized, sample_aff

def registration_deform(reference_contour, aligned_contour, show_callback=True):
    
    reg = pycpd.deformable_registration(X=reference_contour, Y=aligned_contour)
    
    if show_callback:
        sample_reg, _ = reg.register(_callback)
    else:
        sample_reg, _ = reg.register()
    
    return sample_reg

def alignement(reference_contour, sample_contour, show_callback=True):
    reference_standardized, sample_standardized, sample_affine = registration_affine(reference_contour, sample_contour, show_callback)
    sample_reg = registration_deform(reference_standardized, sample_affine)
    return reference_standardized, sample_standardized, sample_affine, sample_reg

def plot_contours(list_contours):
    plt.figure()
    for i in range(len(list_contours)):
        plt.plot(*list_contours[i].T)
    plt.show()
    return