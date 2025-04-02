# Pre-SN Variability

Developed by Tobias Géron at the University of Toronto. Created Summer 2024. Last updated Mar 2025. This code was last tested using the Weekly 2024_42 image on the RSP. Please see below for a full list of package compatibility.

This code was developed to help better quantify detection thresholds with the Rubin pipelines. The current Rubin pipelines define detections with a simple SNR threshold. However, this becomes difficult when there is a bright background galaxy, or when the datapoint in question has a magnitude around the detection threshold (~23-24). How confident are we that this detection is real?

This code was created with a specific science goal in mind: pre-SN variability. However, it can be used in any science context where you want to better understand the detection thresholds. We aim to answer the question: "Using the Rubin pipelines, in this specific epoch, using this specific template, how likely is it that a datapoint of X mag is detected?" 

This is answered using a series of source injection, image subtraction, and forced photometry. The output is a recovery curve, which you can use to compute the 80% or 50% detection fraction thresholds (i.e. the magnitude where X% of sources with magnitude Y would be detected with the Rubin pipelines at this specific epoch and this specific template).



### Minimal example

Below you can find a minimal example of how to use this package. However, it is strongly encouraged to have a look at the more in-depth tutorials as well. 

First, import the package.

```
from pre_sn_variability import *
```

Then, run the main part of the code. This will take some time, depending on the settings. See the in-depth tutorial for more detail on how to speed this up, if needed. You need a magnitude, the science and template exposures, other sources in the image, the SN coordinates, and the config object.

```

config = recovery_curve_config()
config.cutout_size = 300 # Strongly encouraged to add a cutout size.

df_recovery = recovery_curve(sn_mag = mag, science_exposure = calexp, template_exposure = template, sources = sources, sn_position = (ra,dec), config = config)
```

You can then use that output to obtain the X% detection fraction thresholds.

```
thresholds = find_thresholds(df_recovery, detection_fraction_thresholds = [0.5,0.8])
```

Which you can compare to the magnitude of your detection. It is also strongly advised to correct your detection magnitude for any background signal when making the comparison:

```
df_background = estimate_sn_background(calexp, template, sources = sources, sn_position = (ra,dec))
    
sn_mag_corrected = njy_to_mag(mag_to_njy(sn_mag) - np.median(df_background['base_PsfFlux_nJy'])) # Convert SN magnitudes to nJy, then subtract median of the background, then convert back to mag. 
```

Finally, you can then compare `sn_mag_corrected` to the previously established thresholds (in `thresholds`) to determine whether the detection is real. Please refer to the example notebooks for more detail (highly recommended!).



### Description parameters in recovery_curve_config class 

There are a lot of parameters that you can finetune in the `recovery_curve_config` class. Many of them are explained in greater detail in the example notebook. However, here is a brief description of each parameter:

We try to be smart while sampling the recovery curve. We will have multiple rounds of sampling and slowly zoom in to the
region where the curve goes from 1 to 0. The following parameters define this process.
`n_mag_steps`: list of ints. Defines how many mag steps in each iteration. Better to start low, and go up. 
`sampling_buffer`: float. During the first iteration, we create a range around the sn_mag. The buffer defines how far out on each side. 
`mag_limits`: list of floats. If the transition region is not found in the first iteration, we simply set the limits of the search to this.

Parameters related to image smoothing:
`smooth_function`: the function used to smooth
`smooth_filter_size`: int. The size of the filter, in pixels

Related to finding injection locations:
`injection_method`: can be either 'random' or 'smooth'. When random, we just return random pixels in the inmage. When smooth, we first smooth the template image, and then try to find pixels that are similar to the target SN. Very highly recommended to use smooth.
`n_injections`: int or list of ints. The amount of injections at each iteration. The precision on the detection fraction depends on this. E.g. if n_injections = 10, then we will only be precise with steps of 0.1 (i.e. 1/n_injections).
`max_inj_per_round`: int. The maximum number of injections done per injection iteration. 
`p_threshold`: int. We try to find injection sites that are similar to the target SN. Every injection site will be p_threshold percent within the value of the pixel at the SN location in the template image.
`psf_flux_threshold`: float. Used when creating the injection iterations. We do not want to plce injection sites too close together. We look at the PSF of each injection location and find the distance at which 1 - psf_flux_threshold of the flux of the normalized PSF is contained and make sure to exclude other sites within that range.
`psf_sigma`: In addition to the above, we also calculate the std of the PSF and add a psf_sigma multiple of that to this radius. 
`min_dist_across_iterations`: even across injection iterations, we don't simply want to take the next pixel over. So whenever we find a suitable injection location, also mark all the other possible ones that are within X arcsec as unuseable.
`injection_n_attempts`: Since the creation of the final injection locations from all the possible injection locations is a random process, we try a few times, and see if there is a suitable solution.
`max_dist`: (float) the maximum distance, in arcsec, away from the SN that the injection locations can be. Default is 20.
`inject_band`: str. Band to inject in. Default 'g'. 

Remaining parameters:
`subtraction_config`: We use the AlardLuptonSubtractTask to do the subtraction. Note that you can also pass a `config` parameter to `subtract_images()`. This should be a `lsst.ip.diffim.subtractImages.AlardLuptonSubtractConfig`. This allows you to finetune the details of the subtraction. If you do not specify this parameter, we will use the default config. Strongly recommended to keep this to default, unless you are confident in what you want to change. 
`snr_threshold`: float. What SNR threshold defines a detection. Default Rubin value is 5. 
`subtract_background`: bool. Whether or not to automatically subtract the background.
`cutout_size`: int. We recommend to input the full images, but we fill create a cutout ourselves for the majority of the tasks to speed things up. Units are arcsec. Default is NaN, but very highly recommended to use a value. It needs to be high enough to still include other sources for the subtraction. Around 300 seems  usually appropriate.
`sn_position_units`: string. Whether the sn_position is in pixel units or sky units.
`plot`: Whether to automatically output plots.
`expand_output`: bool. If True, we expand the output to contain much more detail. You can recover the summarised dataframe by doing: `df_recovery = recovery_summary(df_expanded_output)`
`n_jobs`: Used to specify how many parallel processes should be used to complete this. Should not exceed number of available CPU cores. Can disable this functionality by doing n_jobs = 0. Parallelisation happens over inject_subtract_photometry() with different sn_mags. 


### How to run this code

You have two options: either run this locally on your device, but then you would need to have the Rubin pipelines installed as well (see https://pipelines.lsst.io/), which might be difficult. It is probably easier to copy this folder to the Rubin Science Platform (https://data.lsst.cloud/), load the appropriate image, and run it on the RSP. It is highly recommended to load a large batch (4 CPU and 16 GB RAM). 



### Package compatibility and versions
You need access to the Rubin pipelines for this to work. Ideally, you would run it directly on the Rubin Science Platform. 

This code was last tested using the Weekly 2024_42 image on the RSP. It is highly recommended to load a large batch (4 CPU and 16 GB RAM). 

In addition, you'll also need:
numpy 1.26.4
matplotlib 3.9.2
tqdm 4.66.5
astropy 6.1.3
pandas 2.2.2
scipy 1.13.1

(Though these are all included in the Weekly 2024_42 image in the right version, so if you're running this code on the RSP using the correct image, don't worry about it.)
