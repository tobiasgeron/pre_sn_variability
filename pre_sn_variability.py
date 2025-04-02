'''
Developed by Tobias Geron at the University of Toronto. Created Summer 2024. Last updated Mar 2025. 
This code was last tested using the Weekly 2024_42 image. 


TODO: Allow users to bin data using coadds. See DP02_09a_Custom_Coadd
'''



###=========###
### Imports ###
###=========###


import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from astropy.coordinates import SkyCoord
import astropy.units as u
import copy
import time
from astropy.table import Table, vstack
import pandas as pd
import math
import random
from scipy.spatial import distance_matrix
from scipy.ndimage import generic_filter, median_filter, uniform_filter
from scipy.optimize import curve_fit
from multiprocessing import Pool
from functools import partial


# Various LSST pipelines
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
from lsst.daf.butler import Butler
import lsst.geom as geom
import lsst.afw.table as afwTable
from lsst.rsp import get_tap_service, retrieve_query


# Source injection
from lsst.source.injection import ingest_injection_catalog, generate_injection_catalog
from lsst.source.injection import VisitInjectConfig, VisitInjectTask
from lsst.daf.butler.registry import ConflictingDefinitionError

# Subtraction
from lsst.ip.diffim.subtractImages import AlardLuptonSubtractTask, AlardLuptonSubtractConfig

# Forced photometry
import lsst.meas.base as measBase





### ================= ###
### General functions ###
### ================= ###

# General functions needed throughout



def pixel_to_sky(coords_pix, wcs):
    '''
    Transforms a list of x,y coordinates to a list of ra,dec using a WCS.
    '''
    single_target = False

    
    if np.array(coords_pix).shape == (2,): #it is a tuple/list of two coordinates. Wrap it in a list
        coords_pix = [coords_pix]
        single_target = True
    
    coords_sky = []
    for l in coords_pix:
        x,y = l[0],l[1]
        coord_sky = (wcs.pixelToSky(x,y)[0].asDegrees(), wcs.pixelToSky(x,y)[1].asDegrees())
        coords_sky.append(coord_sky)

    if single_target:
        return coords_sky[0]
    else:
        return coords_sky



def sky_to_pixel(coords_sky, wcs):
    '''
    Transforms a list of ra,dec (in degrees) coordinates to a list of x,y using a WCS.
    '''
    single_target = False

    if np.array(coords_sky).shape == (2,): # #it is a tuple/list of two coordinates. Wrap it in a list
        coords_sky = [coords_sky]
        single_target = True
    
    coords_pix = []
    for i in coords_sky:
        ra, dec = i[0], i[1]
        coord_sky = geom.SpherePoint(ra*geom.degrees, dec*geom.degrees)
        coord_pix = (wcs.skyToPixel(coord_sky)[0],wcs.skyToPixel(coord_sky)[1])
        coords_pix.append(coord_pix)

    if single_target:
        return coords_pix[0]
    else:
        return coords_pix







def add_noise(exp, mu = 0, stdev = 1, method = 'additive'):
    '''
    Adds noise to an image. Can either be additive or multiplicative. Used for debugging purposes.
    '''
    shape = exp.image.array.shape
    noise = np.random.normal(loc=mu, scale=stdev, size=shape)
    
    if method == 'additive':
        exp.image.array += noise


    elif method == 'multiplicative':
        exp.image.array *= noise
        
    return exp



def smooth_image(exposure, function = 'median', filter_size = 50):
    '''
    This will smoothen an image. 

    Passing function = 'median' or 'mean' will use the scipy optimised median_filter and uniform_filter. This is 
    much faster than using np.median. 
    '''
    exposure_temp = exposure.clone() #we don't want to overwrite the original exposure
    img = exposure_temp.image.array

    
    if function == 'median':
        smoothened = median_filter(img, size=filter_size)
    elif function == 'mean':
        smoothened = uniform_filter(img, size=filter_size)
    else:
        smoothened = generic_filter(img, function = function, size=filter_size)
    

    exposure_temp.image.array = smoothened
    return exposure_temp
    




def union(list1,list2):
    '''
    Calculates union of two lists
    '''
    union_list = list(set(list1).union(set(list2)))
    return union_list


def intersection(list1, list2):
    '''
    Calculates intersection of two lists
    '''
    intersection_list = list(set(list1).intersection(set(list2)))
    return intersection_list




def plot_image(exposure, fig = '', ax = '', scale = 'asinh', vmin = np.nan, vmax = np.nan, vmin_p = 10, vmax_p = 98, title = '', plot_ticks = True, plot_colorbar = True, coords = [], coords_cols = [], zoom_target = [], zoom_size = 40, fontsize = 12):
    """
    image should be a lsst.afw.image._image.ImageF
    You can specificy fig and ax, if you wanted to. Otherwise it'll use the current active ones.
    You can set the vmin and vmax, but otherwise it'll calculate the percentiles specificied at vmin_p and vmax_p

    You can add a list of ra,dec pairs to coords, which this will plot as a scatter plot.

    zoom_size is in arcsec
    """


    pixscale = exposure.getWcs().getPixelScale().asArcseconds()
    zoom_size = int(zoom_size / pixscale)
    

    if zoom_target != []:
        ra,dec = zoom_target
        exposure = cutout_exposure(exposure, ra, dec, size = zoom_size)

        
    image = exposure.image
    wcs = exposure.getWcs()
    
    
    if fig == '':
        fig = plt.gcf()
    if ax == '':
        ax = plt.gca()

    if np.array(coords).shape == (2,): # #it is a tuple/list of two coordinates. Wrap it in a list
        coords = [coords]


    if np.isnan(vmin):
        vmin = np.nanpercentile(image.array, vmin_p)
    if np.isnan(vmax):
        vmax = np.nanpercentile(image.array, vmax_p)



    plt.sca(ax)
    display0 = afwDisplay.Display(frame=fig)
    display0.scale(scale, min=vmin, max=vmax)
    display0.mtv(image)

    if plot_colorbar == False:
        pass
        #display0.setColorBar(None)
    
    if title != '':
        plt.title(title, fontsize = fontsize)

    if plot_ticks == False:
        plt.xticks([])
        plt.yticks([])


    if coords_cols == []:
        coords_cols = [f'C{i}' for i in range(len(coords))]
    for i,coord in enumerate(coords):
        ra,dec = coord
        sky_coord = geom.SpherePoint(ra*geom.degrees, dec*geom.degrees)
        pixel_coord = wcs.skyToPixel(sky_coord)
        plt.scatter(pixel_coord[0], pixel_coord[1], marker = 'o', edgecolor = coords_cols[i], facecolor = 'none', s = 100)
        



def plot_image_v2(exposure, fig = '', ax = '', scale = 'asinh', vmin = np.nan, vmax = np.nan, vmin_p = 10, vmax_p = 99.9999, title = '', plot_ticks = True, plot_colorbar = True, coords = [], coords_cols = [], coords_size = [], zoom_target = [], zoom_size = 40, cmap = 'gray', fontsize = 12, plot_contour = False, contour_levels = 0, contour_smooth_size = 50, contour_alpha = 1):
    """
    Update to plot_image to use plt.imshow instead of the built-in awfDisplay to allow for more customisation.
    """

    assert scale in ['asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog'], ''

    pixscale = exposure.getWcs().getPixelScale().asArcseconds()
    zoom_size = int(zoom_size / pixscale)


    if zoom_target != []:
        ra,dec = zoom_target
        exposure = cutout_exposure(exposure, ra, dec, size = zoom_size)

        
    image = exposure.image
    wcs = exposure.getWcs()
    
    
    if fig == '':
        fig = plt.gcf()
    if ax == '':
        ax = plt.gca()

    if np.array(coords).shape == (2,): # #it is a tuple/list of two coordinates. Wrap it in a list
        coords = [coords]


    if np.isnan(vmin):
        vmin = np.nanpercentile(image.array, vmin_p)
    if np.isnan(vmax):
        vmax = np.nanpercentile(image.array, vmax_p)

    exposure_extent = (exposure.getBBox().beginX, exposure.getBBox().endX,
                 exposure.getBBox().beginY, exposure.getBBox().endY)
    
    
    plt.sca(ax)
    
    plt.imshow(exposure.image.array, origin = 'lower', norm =scale, cmap = cmap, vmin = vmin, vmax = vmax, extent = exposure_extent)
    
    if title != '':
        plt.title(title, fontsize = fontsize)

    if plot_ticks == False:
        plt.xticks([])
        plt.yticks([])

    if plot_colorbar:
        plt.colorbar()

    if coords_cols == []:
        coords_cols = [f'C{i}' for i in range(len(coords))]
    if coords_size == []:
        coords_size = [100] * len(coords)
    for i,coord in enumerate(coords):
        ra,dec = coord
        sky_coord = geom.SpherePoint(ra*geom.degrees, dec*geom.degrees)
        pixel_coord = wcs.skyToPixel(sky_coord)
        plt.scatter(pixel_coord[0], pixel_coord[1], marker = 'o', edgecolor = coords_cols[i], facecolor = 'none', s = coords_size[i])
        


    if plot_contour:
        exposure_smooth = smooth_image(exposure, function = 'median', filter_size = contour_smooth_size)
        #exposure_smooth = exposure
        
        if contour_levels == 0:
            start = 1 * np.nanstd(exposure_smooth.image.array.flatten())
            end = np.nanmax(exposure_smooth.image.array.flatten())
            contour_levels = [start]
            while start < end:
                start = start*np.sqrt(2)
                contour_levels.append(start)
            
        plt.contour(exposure_smooth.image.array, levels = contour_levels, extent = exposure_extent, colors = 'white', alpha = contour_alpha)  # Adjust levels for detail




    

def cutout_exposure(exposure, ra, dec, size = 101, size_units = 'pixel'):
    '''
    Will create a square cutout with a specific in at exposure around ra/dec value 
    size is in pixels by default, but can change that using size_units.
    '''

    wcs = exposure.getWcs()
    
    assert exposure.width >= size, f"The size of the full exposure {exposure.width} is smaller than the desired cutout."
    assert exposure.height >= size, f"The size of the full exposure {exposure.height} is smaller than the desired cutout."
    assert size_units in ['sky','pixel'], "size_units can either be 'sky' or 'pixel'"


    # convert to pixel units if needed
    if size_units == 'sky':
        pixscale = exposure.getWcs().getPixelScale().asArcseconds()
        size = int(size / pixscale)
    
    sky_position = geom.SpherePoint(ra, dec, geom.degrees)
    pix_position = geom.PointI(wcs.skyToPixel(sky_position))

    xmin, xmax = pix_position[0] - size //2, pix_position[0] + size //2
    ymin, ymax = pix_position[1] - size //2, pix_position[1] + size //2

    b = exposure.getBBox()
    X0, Y0, X1, Y1 = b.beginX, b.beginY, b.endX, b.endY


    if xmin < X0:
        xmax += np.abs(xmin)
        xmin = X0

    if ymin < Y0:
        ymax += np.abs(ymin)
        ymin = Y0

    if xmax > X1:
        xmin -= (np.abs(xmax - X1))
        xmax = X1

    if ymax > Y1:
        ymin -= (np.abs(ymax - Y1))
        ymax = Y1

    box2i = geom.Box2I(geom.IntervalI(min = xmin, max = xmax-1), 
                       geom.IntervalI(min = ymin, max = ymax-1))
    cutout = exposure.getCutout(box2i)

    return cutout


def cutout_butler(butler, dataId, ra, dec, dataset_type = 'calexp', size=101):
    
    """
    Instead of returning the whole calexp, return a cutout. 


    NOTE: img.width corresponds to x-axis, img.height to y-axis. 
    """
        
    full = butler.get(dataset_type, dataId = dataId)
    cutout = cutout_exposure(full, ra, dec, size)

    return cutout




def filter_sources(sources, cutout, buffer = 2):
    # Only include sources that are within a cutout.

    schema = sources.schema
    sources_temp = afwTable.SourceCatalog(schema)
    
    cutout_shape = cutout.height, cutout.width
    xmin, ymin = cutout.image.getX0(), cutout.image.getY0()
    xmax = xmin + cutout_shape[1]
    ymax = ymin + cutout_shape[0]

    
    wcs = cutout.getWcs()
    for i,s in enumerate(sources):
        sky_coord = geom.SpherePoint(s['coord_ra'].asDegrees()*geom.degrees, s['coord_dec'].asDegrees()*geom.degrees)
        pixel_coord = wcs.skyToPixel(sky_coord)
        x_pixel, y_pixel = pixel_coord
        in_bounds = (xmin + buffer < x_pixel < xmax - buffer) and (ymin + buffer < y_pixel < ymax - buffer)
    
        if in_bounds:
            sources_temp.append(s)  # Append the rows you want to keep
    
    sources = sources_temp.copy(deep=True)
    return sources



def in_exposure(sky_coord, cutout, buffer = 0):
    cutout_shape = cutout.height, cutout.width
    xmin, ymin = cutout.image.getX0(), cutout.image.getY0()
    xmax = xmin + cutout_shape[1]
    ymax = ymin + cutout_shape[0]

    wcs = cutout.getWcs()
    pixel_coord = sky_to_pixel(sky_coord, wcs = wcs)
    x_pixel, y_pixel = pixel_coord

    in_bounds = (xmin + buffer < x_pixel < xmax - buffer) and (ymin + buffer < y_pixel < ymax - buffer)
    return in_bounds
    
    
    



### ========= ###
### Injection ###
### ========= ###

# Functions related to source injection. Both doing the injection, and finding the injection locations

def create_injection_catalog(ra, dec, source_type, mag, n = [], q = [], 
                             half_light_radius = [], beta = [], stamp_loc = [],
                             rotation = []):
    '''
    This def is good if you want to inject specific galaxies or stars
    If you want randomised galaxies, perhaps better to use the more general 
    LSST one: `generate_injection_catalog()`.

    Currently only supporting source_type in ['Sersic','Star','Stamp'].

    Every parameter is a list. 

    Note: stamps need to be fits files with accurate wcs. If you add a rotation, this code will first 
    open the original fits image, rotate it, save it as a fits file, and then change the stamp_loc to 
    the rotated file.
    '''

    # Sersic galaxies
    ind_galaxy = np.where(np.array(source_type) == 'Sersic')[0]

    t_galaxy = Table([np.array(ra)[ind_galaxy], np.array(dec)[ind_galaxy], 
                      np.array(source_type)[ind_galaxy], np.array(mag)[ind_galaxy], 
                      np.array(n)[ind_galaxy], np.array(q)[ind_galaxy], 
                      np.array(half_light_radius)[ind_galaxy], np.array(beta)[ind_galaxy]],
     names = ('ra','dec','source_type','mag','n','q','half_light_radius','beta'))



    # Stars
    ind_star = np.where(np.array(source_type) == 'Star')[0]

    t_star = Table([np.array(ra)[ind_star], np.array(dec)[ind_star], 
                      np.array(source_type)[ind_star], np.array(mag)[ind_star]],
     names = ('ra','dec','source_type','mag'))


    # Stamps
    ind_stamp = np.where(np.array(source_type) == 'Stamp')[0]

    # Deal with rotating
    if rotation == []:
        rotation = [0] * len(source_type)

    for idx_stamp in ind_stamp: #Go over stamps, see if they need to be rotated. 
        if rotation[idx_stamp] != 0:
            stamp_img_orig = afwImage.ExposureF.readFits(stamp_loc[idx_stamp])
            rotation_angle = rotation[idx_stamp] 
            stamp_img_rotated = rotate_exposure(stamp_img_orig, rotation_angle) # rotate image
            stamp_img_rotated.image.array[np.where(np.isnan(stamp_img_rotated.image.array))] = 0.0 # Set all NaNs to 0
            stamp_path_rotated = f"{stamp_loc[idx_stamp].replace('.fits','')}_rotated_{rotation_angle}.fits"
            stamp_img_rotated.writeFits(stamp_path_rotated)
            stamp_loc[idx_stamp] = stamp_path_rotated
            
    
    t_stamp = Table([np.array(ra)[ind_stamp], np.array(dec)[ind_stamp], 
                      np.array(source_type)[ind_stamp], np.array(mag)[ind_stamp],
                    np.array(stamp_loc)[ind_stamp]],
     names = ('ra','dec','source_type','mag','stamp'))


    t = vstack([t_galaxy, t_star, t_stamp])
    return t




def inject_source(image, injection_catalog, band = 'g', butler_config = 'dp02', butler_collections = '2.2i/runs/DP0.2',
                  injection_catalog_name = 'injection_catalog', log_level = 30):
    '''
    Will inject a source catalog into an image. Automatically handles the ingestion of the catalog, reloading, task configuration and running. 


    Updates
    2024-11-08: Noticed that you don't have to ingest and reload injection catalogs. 
    '''


    
    """ 
    ### Step 1: Ingest the injection catalog
    user = os.getenv("USER")
    INJECTION_CATALOG_COLLECTION = f"u/{user}/{injection_catalog_name}_{int(time.time())}_{random.randint(0,10000)}" #_{random.randint(0,100)} is for when you're running multiple notebooks at the same time
    writeable_butler = Butler(butler_config, writeable=True)
    
    try:
        my_injected_datasetRefs = ingest_injection_catalog(
            writeable_butler=writeable_butler,
            table=injection_catalog,
            band=band,
            output_collection=INJECTION_CATALOG_COLLECTION,
            log_level = log_level
        )
    except ConflictingDefinitionError:
        print(f"Found an existing collection named INJECTION_CATALOG_COLLECTION={INJECTION_CATALOG_COLLECTION}.")
        print("\nNOTE THAT IF YOU SEE THIS MESSAGE, YOUR CATALOG WAS NOT INGESTED."\
              "\nYou may either continue with the pre-existing catalog, or choose a new"\
              " name and re-run the previous cell and this one to ingest a new catalog.")


    ### Step 2: Load input injection catalogs.
    butler = Butler(butler_config, collections=butler_collections)
    injection_refs = butler.registry.queryDatasets(
        "injection_catalog",
        band=band,
        collections=INJECTION_CATALOG_COLLECTION,
    )
    injection_catalog = [
        butler.get(injection_ref) for injection_ref in injection_refs
    ]

    """

    ### Step 3: Configure Task

    inject_config = VisitInjectConfig()
    #inject_config.stamp_prefix='data/' I don't think we need this? 
    inject_task = VisitInjectTask(config=inject_config)
    
    ### Step 4: Run Task
    psf = image.getPsf()
    photo_calib = image.getPhotoCalib()
    wcs = image.getWcs()
    
    injected_output = inject_task.run(
        injection_catalogs=injection_catalog,
        input_exposure=image.clone(),
        psf=psf,
        photo_calib=photo_calib,
        wcs=wcs,
    )

    ### Step 5: return
    injected_exposure = injected_output.output_exposure
    injected_catalog = injected_output.output_catalog
    return injected_exposure, injected_catalog






def find_random_injection_locations(n_injections, exposure, output_units = 'pixel', sn_max_dist = 0, sn_loc = []):
    '''
    Will create injection locations at random locations. 
    exposure should be lsst.afw.image._exposure.ExposureF

    Can also provide location of target and a maximum distance from it.
    sn_max_dist should be in arcsec. 
    sn_loc in ra/dec (degrees).
    '''

    assert output_units in ['pixel','sky'], "output units can either be pixel or sky"

    
    size_image = exposure.image.array.shape
    xmin, ymin = exposure.getXY0()
    xmax, ymax = size_image[1]+xmin, size_image[0]+ymin

    if sn_max_dist != 0 and len(sn_loc) != 0:
        x_sn, y_sn = sky_to_pixel(sn_loc, wcs = exposure.getWcs())
        pixscale = exposure.getWcs().getPixelScale().asArcseconds()

        
        xmin_sn, ymin_sn = x_sn - sn_max_dist/pixscale/2, y_sn - sn_max_dist/pixscale/2
        xmax_sn, ymax_sn = x_sn + sn_max_dist/pixscale/2, y_sn + sn_max_dist/pixscale/2

        xmin = int(np.max([xmin, xmin_sn]))
        ymin = int(np.max([ymin, ymin_sn]))

        xmax = int(np.min([xmax, xmax_sn]))
        ymax = int(np.min([ymax, ymax_sn]))
        
    lst = []
    for i in range(n_injections):
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        lst.append((x,y))

    if output_units == 'sky':
        lst = pixel_to_sky(lst, wcs = exposure.getWcs())
    
    return lst




def find_smoothened_injection_locations(exposure, coor, smooth_function = 'median', smooth_filter_size = 10, n_injections = 0, p_threshold = np.inf, output_units = 'pixel'):
    '''
    Will return closest coordinates in an image based on coor. Input needs to be ra/dec.
    Will output a list of (x,y) coordinates or ra/dec, depending on output_pixels. 
    Can specify either to return the n closest locations, or all locations that are p percent within the value of coor. 
    Can also combine. E.g.: if n_injections = 10 and p_threshold = 5, this will return 10 coordinate pairs, and guarantee they are all within 5% of the 
    original value. If only 6 have values that are <5%, only these 6 will be returned.

    This smoothens the exposure first. Ideally you want the smoothening to get rid of stars in the picture. If you don't want any smoothening, do smooth_filter_size = 0.
    '''
    assert output_units in ['pixel','sky'], "output units can either be pixel or sky"
    
    if smooth_filter_size > 0:
        smoothened_exposure = smooth_image(exposure, function = smooth_function, filter_size = smooth_filter_size)
    else:
        smoothened_exposure = exposure.clone()

    
    coor = sky_to_pixel(coor, wcs = exposure.getWcs())
    idxs = find_closest_values(smoothened_exposure, coor, n_injections = n_injections, p_threshold = p_threshold)


    if output_units == 'sky':
        idxs = pixel_to_sky(idxs, wcs = exposure.getWcs())

        
    return idxs, smoothened_exposure



def create_injection_locations(science_exposure, n_injections, max_per_round = 100, sn_position = [], template_exposure = [], method = 'smooth', plot = False, plot_zoom = False, 
                              plot_zoom_size = 40, smooth_function = 'median', smooth_filter_size = 10, p_threshold = 5, psf_flux_threshold = 0.0001, psf_sigma = 3,
                              max_dist = 20, radius_increment = 2, min_dist_across_iterations = 1, n_attempts = 10):

    '''
    sn_position must be in ra/dec

    max_dist is maximum distance of injection locations to SN in arcsec

    min_dist_across_iterations: even across injection iterations, we don't simply want to take the next pixel over. So whenever we find a suitable injection location, also mark all the other possible ones that are within X arcsec as unuseable.

    TODO: make separate arguments for minimum distance to SN?
    '''


    ### Step 1: find possible locations based on method
    assert method in ['random','smooth'], "method has to be either 'random' or 'smooth'"

    if method == 'random':
        injection_locations = find_random_injection_locations(n_injections*20, science_exposure, output_units = 'sky')
        
    elif method == 'smooth':
        assert sn_position != [] and template_exposure != [],"Please provide a SN location and a template exposure to use the smooth method"
        injection_locations, smoothened_exposure = find_smoothened_injection_locations(template_exposure, sn_position, smooth_function = smooth_function, smooth_filter_size = smooth_filter_size, p_threshold = p_threshold, output_units = 'sky')
        
        
        assert len(injection_locations) >= n_injections, "Couldn't find enough possible injection locations. Please incresase p_threshold."


    ### Step 2: Filter out locations that are too close to the current SN
    min_dists = create_min_dists(science_exposure, injection_locations, psf_flux_threshold = psf_flux_threshold, psf_sigma = psf_sigma, radius_increment = radius_increment)


    if sn_position != []: # Also filter out any injection locations that are too close (or too far away) to SN position

        sn_min_dist = create_min_dists(science_exposure, [sn_position], psf_flux_threshold = psf_flux_threshold, psf_sigma = psf_sigma, radius_increment = radius_increment)
        injection_locations, min_dists = filter_injection_locations(injection_locations,min_dists,[sn_position], sn_min_dist, remove_max_dists = [max_dist/3600])


    assert len(injection_locations) >= n_injections, "Couldn't find enough possible injection locations. Please relax criteria by e.g. increasing p_threshold if method = smooth."


    
    ### Step 3: Find good combination of locations to inject together.
    # Since it is a random process, try a few times, see if it works. 
    found_injection_locations = False
    for i_attempt in range(n_attempts):
        try:
            injection_locations = create_injection_list(injection_locations, min_dists, n_injections, max_per_round = max_per_round, min_dist_across_iterations = min_dist_across_iterations)
            found_injection_locations = True
            break
        except:
            pass

    assert found_injection_locations, "Couldn't find enough possible injection locations. Please relax criteria by e.g. decreasing min_dist_across_iterations or increasing p_threshold if method = smooth."
        

    ### Step 4: Plot, if so desired. Good for debugging purposes
    if plot:
        if plot_zoom == False:
            b = science_exposure.getBBox()
            zoom_target = (b.centerX, b.centerY)
            zoom_target = pixel_to_sky(zoom_target, wcs = science_exposure.getWcs())
            plot_zoom_size = math.floor(science_exposure.width * science_exposure.getWcs().getPixelScale().asArcseconds())
        else:
            zoom_target = sn_position

        
        nrow = 1
        ncol = 3 if method == 'smooth' else 2
        i_plot = 0
        
        vmin, vmax = np.nanpercentile(science_exposure.image.array,10),np.nanpercentile(science_exposure.image.array,99.8)

        fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*4,3*nrow))

        plot_image(science_exposure, ax = ax[i_plot], title = 'Raw science exposure', zoom_target = zoom_target, zoom_size = plot_zoom_size)
        i_plot += 1

        if method == 'smooth':
            plot_image(smoothened_exposure, ax = ax[i_plot], title = 'Smoothened template exposure', zoom_target = zoom_target, zoom_size = plot_zoom_size)
            i_plot += 1

        plot_image(science_exposure, ax = ax[i_plot], title = 'Injection locations', zoom_target = zoom_target, zoom_size = plot_zoom_size)

        
        cols = [f'C{i}' for i in range(len(injection_locations))]
        for i,idxs in enumerate(injection_locations):
            for idx in idxs:
                idx_pix = sky_to_pixel(idx, wcs = science_exposure.getWcs())
                plt.scatter(idx_pix[0], idx_pix[1], facecolor = 'none', edgecolor = cols[i], s = 100)

        sn_position_pix = sky_to_pixel(sn_position, wcs = science_exposure.getWcs())
        plt.scatter(sn_position_pix[0],sn_position_pix[1], facecolor = 'none', edgecolor = 'red', s = 100, marker = 'X')

        plt.tight_layout()
        plt.show()
    

    return injection_locations
    
    
    


def lower_injection_locations(injection_locations, n):
    '''
    Will remove n injection locations from an injection_locations list of lists.
    '''

    injection_locations_temp = copy.deepcopy(injection_locations)
    for i in range(n):
        injection_locations_temp[-1].pop(-1)
        if len(injection_locations_temp[-1]) == 0:
            injection_locations_temp.pop(-1)
    return injection_locations_temp





def filter_injection_locations(locs, min_dists, remove_locs, remove_min_dists, remove_max_dists):
    '''
    Remove any possible injection sites if they are too close to any specific positions. In practise that would be when they are 
    too close to the original SN.
    '''

    keep = [True] * len(locs)
    dist_matrix = distance_matrix(locs, remove_locs)

    for i,l in enumerate(locs): #Go over every possible position
        for j, rl in enumerate(remove_locs):
            dist = dist_matrix[i][j]
            if dist <= min_dists[i] or dist <= remove_min_dists[j]:
                keep[i] = False
            if dist > remove_max_dists[j]:
                keep[i] = False

    return np.array(locs)[keep], np.array(min_dists)[keep]





def create_injection_list(locs, min_dists, n_injection, max_per_round = 1000, min_dist_across_iterations = 1):
    '''
    If we inject locations one-by-one, then algorithm takes forever. Here, we will identify locations that you 
    are able to inject together
    
    NOTE: this is not optimised. It is not the best solution or combination of injection locations. 
    It is a quick, possible solution to speed things up a bit.

    locs: list of possible injection sites, in pixel units
    min_dists: list of how far away each injection site has to be from others
    n_injection: how many to select in total
    max_per_round: how many injections to maximum put in one round. Can set this to 1 if you want to disable this feature
    min_dist_across_iterations: even across injection iterations, we don't simply want to take the next pixel over. So whenever we find a suitable injection location, also mark all the other possible ones that are within X arcsec as unuseable.


    Def will return a list of lists. Every sublist will be a x,y pair that can be injected together.

    TODO:
    instead of taking random one, preferentially take one far away? Well, that becomes confusing for multiple
    targets. Could do a sum of all of their cumulative distances, or something. Well, not cumulative, but 
    multiplicative. That was it is better to be far away from all others currently using. 

    TODO: the naming of variables here is not great. Make it better
    '''

    

    assert len(locs) >= n_injection, "The amount of possible injection sites cannot be smaller than the amount of desired injection sites"
    assert len(locs) == len(min_dists), "The list with the injection locations must be the same length as the list with their minimum distances"

    
    used = np.array([False] * len(locs)) #To track which ones we've already used
    dist_matrix = distance_matrix(locs, locs)

    
    # Init
    n_injection_found = 0
    injection_idxs = []
    
    # Keep looping until we've found enough injection sites. Worst case scenario there is only one site found per loop.
    # Every loop here will be a separate collection of injection locations that do not overlap
    
    while n_injection_found < n_injection:
        
        max_locs_needed = n_injection - n_injection_found # max locs needed in this iteration
        
        temp = []
    
        # First addition. Always add one
        idxs = np.where(used == False)[0]
        idx = random.choice(idxs)
        used[idx] = True
        temp.append(idx)
        idxs = np.where(dist_matrix[idx] < min_dist_across_iterations/3600)[0] #Also set all possible locations that are within X arcsec. Even across injection iterations, we don't want exact overlap.
        used[idxs] = True

        
        # Select next possible additions
        # Do over while loop as long as there are possible new targets that are
        # Sufficiently far away from any already added this iteration and not 
        # used before.
        still_possible_targets = True
        while still_possible_targets and len(temp) < max_locs_needed and len(temp) < max_per_round:
            possibilities = list(range(len(locs)))
            for idx in temp:
                possible = np.where( (dist_matrix[idx] > min_dists[idx]) & (used == False))[0]
                possible_temp = []
                for p in possible: #Check if the reverse is also good.
                    if dist_matrix[p][idx] > min_dists[p]:
                        possible_temp.append(p)

                    
                possibilities = intersection(possibilities, possible_temp)
    
            
            if len(possibilities) > 0: #If there still is an option, add it
                idx = random.choice(possibilities)
                used[idx] = True
                temp.append(idx)
                idxs = np.where(dist_matrix[idx] < min_dist_across_iterations/3600)[0] #Also set all possible locations that are within X arcsec. Even across injection iterations, we don't want exact overlap.
                used[idxs] = True
            if len(possibilities) == 0: # If not, exit while loop
                still_possible_targets = False
    
    
        n_injection_found += len(temp)
        injection_idxs.append(temp)
    
    
    
    # Change idx to locs
    injection_locs = []
    for i in injection_idxs:
        temp = []
        for j in i:
            temp.append((locs[j][0], locs[j][1]))
        injection_locs.append(temp)
    return injection_locs


    

def find_closest_values(exposure, coor, n_injections = 0, p_threshold = np.inf):
    '''
    Will return closest pixel coordinates in an image based on coor. Will be a list of (x,y) coordinates. Can specify either to return the n closest pixels, or all pixels 
    that are p percent within the value of coor. Can also combine. E.g.: if n_injections = 10 and p_threshold = 5, this will return 10 coordinate pairs, and guarantee they are all within 5% of the 
    original value. If only 6 have values that are <5%, only these 6 will be returned.
    
    NOTE: When directly accessing image data, indexing is based on y first, then x. 
    
    
    '''

    assert n_injections != 0 or p_threshold != np.inf, "Either set n_injections or p_threshold."

    xmin, ymin = exposure.getXY0()
    img = exposure.image.array.copy()

    if n_injections == 0:
        n_injections = len(img.flatten())-1

    img_min = np.min(img)
    img -= img_min#to make all values positive
    img += 0.001#to avoid dividing by 0

    

    val = img[int(coor[1]-ymin)][int(coor[0]-xmin)]

    diff = np.abs(img - val) / img * 100
    diff[int(coor[1]-ymin)][int(coor[0]-xmin)] = np.inf #just so that it won't be selected itself

    threshold = sorted(diff.flatten())[n_injections]
    
    idxs = np.where((diff < threshold) & (diff < p_threshold))
 
    #idxs = np.array(idxs).T
    idxs = list(zip(idxs[1] + xmin, idxs[0] + ymin))

    return idxs






def get_psf_flux_fraction(exp, location, radius):
    '''
    input in x,y 
    
    Returns what fraction of the flux is covered by the PSF of an ExposureF at a given 
    location and a given sigma size
    '''

    inf_exp = exp.getInfo()
    psf_exp = exp.getPsf()

    point = geom.Point2D((location[0],location[1]))
    #psf_shape = psf_exp.computeShape(point)
    #psf_sigma = psf_shape.getDeterminantRadius()
    
    frac_flux = psf_exp.computeApertureFlux(radius=radius, position=point)
    return frac_flux



def get_psf_radius(exp, location):
    '''
    Returns radius of PSF of an ExposureF at a given location. 

    input in x,y

    When PSF is isotropic (i.e. circularly symmetric), the Determinant Radius should equal sigma.
    ''' 
    inf_exp = exp.getInfo()
    psf_exp = exp.getPsf()

    point = geom.Point2D((location[0],location[1]))
    psf_shape = psf_exp.computeShape(point)
    psf_sigma = psf_shape.getDeterminantRadius()

    return psf_sigma


def find_separation_radius(exposure, point, psf_flux_threshold = 0.001, radius_increment = 2):
    '''
    At a point in an ExposureF, will look at PSF and try to find the radius at which
    1-psf_flux_threshold of the flux is contained. E.g.: psf_flux_threshold = 0.001, 
    it'll find the radius at which 99.9% of the flux is found

    Input needs to be in x,y
    '''

    radius_pix = 1
    
    psf_flux_contained = get_psf_flux_fraction(exposure, point, radius = radius_pix)
    while psf_flux_contained < 1 - psf_flux_threshold:
        radius_pix += radius_increment
        psf_flux_contained = get_psf_flux_fraction(exposure, point, radius = radius_pix)
        

    return radius_pix
    


def create_min_dists(exp, locs, psf_flux_threshold = 0.0001, psf_sigma = 3, radius_increment = 2):
    '''
    Input is in ra/dec.
    
    For all points in locs, find what the minimal distance that it can be to another point, based on its PSF.
    
    For every location, look at PSF at that location, calculate its radius, 
    and multiply it with sigma to get X*sigma distance. 

    Two methods to determine minimum distance. 1) Look at what fraction of the flux is contained within a radius. 
    Finetune that radius so that `psf_flux_threshold` (i.e. 0.01%) of the flux of the normalised psf is contained 
    within that radius. Or 2) determine the sigma of the PSF (well, the determinant radius). Then the minimum distance is
    that radius times psf_sigma (typically at least 10-20, but can be any).

    Will use method 1 by default, unless psf_sigma is specified.

    Output will be in degrees.

    '''
    locs = sky_to_pixel(locs, exp.getWcs()) #transform sky to pixel coordinates

    
    min_dist_pix = []
    for loc in locs:

        if psf_sigma != 0:
            s = get_psf_radius(exp, loc)
            #min_dist_pix.append(psf_sigma*s)
        else:
            s = 0
        
        if psf_flux_threshold != 1:
            rad = find_separation_radius(exp, loc, psf_flux_threshold = psf_flux_threshold, radius_increment = radius_increment)
            #min_dist_pix.append(rad)
        else:
            rad = 0

        min_dist_pix.append(rad+psf_sigma*s)

    
    min_dist_deg = []
    for i in min_dist_pix: #change to degrees
        min_dist_deg.append(exp.getWcs().getPixelScale().asDegrees() * i)
    return min_dist_deg







### =========== ###
### Subtraction ###
### =========== ###


def subtract_images(template_exp, science_exp, sources, subtraction_config = None):
    '''
    template_exp should be an lsst.afw.image._exposure.ExposureF of 
    the template image. 
    science_exp should be an lsst.afw.image._exposure.ExposureF of 
    the science image (e.g. containing a SNe)
    sources should be an lsst.afw.table.SourceCatalog containing sources
    in the image. The AlardLuptonSubtractTask uses it to match. 

    Note from the DP02_14 tutorial:
    In w_2024_38, the source selector within AlardLuptonSubtractTask was changed. This new selector uses 
    a value that is not included by default in the DP0.2 catalogs, so need to add it manually. Do it in 
    a try:except block to avoid errors. 
    '''

    # Configure
    if subtraction_config == None:
        config = AlardLuptonSubtractConfig()
    else:
        config = subtraction_config

        
    try:
        config.sourceSelector.value.unresolved.name = 'base_ClassificationExtendedness_value'
    except:
        pass
    alTask = AlardLuptonSubtractTask(config=config)

    
    # Run
    diff = alTask.run(template_exp, science_exp, sources)

    return diff









### ================= ###
### Forced photometry ###
### ================= ###


def forced_photometry(exposure, coords, return_full = False):
    '''
    Does forced photometry on a ra/dec on an exposure.
    Exposure currently should be a DP0.2 ExposureF. 
    Coords should be a tuple of ra,dec, or list of tuples with ra,decs.  Ra and dec in degrees.
    We only return the important columns. If you want the full dataframe, set return_full = True


    We're using the ForcedMeasurementTask (https://pipelines.lsst.io/py-api/lsst.meas.base.ForcedMeasurementTask.html#lsst.meas.base.ForcedMeasurementTask)

    The run method needs the following arguments:  
    measCat: seems to be an empty catalogue that the function will populate? Not sure.  
    exposure: the actual image  
    refCat: information of the sources. RA/Dec etc.  
    refWcs:  Wcs information, obtained from exposure.


    See notebook tutorial 5 section 5. 
    '''

    if type(coords) == tuple: 
        coords = [coords]

    ### Step 1: Create and configure catalogue with all the sources to do forced photometry on (refCat) ###
 
    schema = afwTable.SourceTable.makeMinimalSchema() # Create a SourceTable Schema, which can be transformed to a Source Catalog

    # The necessary columns might change in the future - if it errors double check that first. 
    alias = schema.getAliasMap() 
    x_key = schema.addField("centroid_x", type="D") # Add centroids columns to schema, ForcedMeasurementTask needs these columns to run
    y_key = schema.addField("centroid_y", type="D")
    alias.set("slot_Centroid", "centroid")

    xx_key = schema.addField("shape_xx", type="D") # Add shape columns schema, ForcedMeasurementTask needs these columns to run
    yy_key = schema.addField("shape_yy", type="D")
    xy_key = schema.addField("shape_xy", type="D")
    alias.set("slot_Shape", "shape")

    sourceCat = afwTable.SourceCatalog(schema) #Change schema into sourceCatalog
    
    for i in range(len(coords)): # Add a new source record for every entry
        ra, dec = coords[i][0], coords[i][1]
        sourceRec = sourceCat.addNew() 
        coord = geom.SpherePoint(ra*geom.degrees, dec*geom.degrees)
        sourceRec.setCoord(coord) 
        
        point = exposure.getWcs().skyToPixel(coord)
        sourceRec[x_key] = point.getX()
        sourceRec[y_key] = point.getY()
    

    ### Step 2: Configure forced photometry and create measCat ###

    # Exact configuration (the plugins and doReplaceWithNoise) might also still change in the future
    # If code errors in the future, check this first.
    #measurement_config = measBase.ForcedMeasurementConfig()
    measurement_config = measBase.ForcedMeasurementTask.ConfigClass()
    measurement_config.copyColumns  = {}  # deblend_nChild is being requested because it's in copyColumns.
    measurement_config.plugins.names = [ # Run only a very minimal set of measurements to avoid complications. Note to self: can also add 'base_CircularApertureFlux' as plugin, if we want circular aperture rather than PSF. 
        "base_TransformedCentroid",
        "base_PsfFlux",
        "base_TransformedShape" 
    ] #"base_PixelFlags",
    measurement_config.doReplaceWithNoise = False #Disable so that we don't need footprints
    #measurement_config.slots.shape = None #To avoid ValueError: "source shape slot algorithm 'base_TransformedShape' is not being run.", but can also just run that plugin instead
    measurement = measBase.ForcedMeasurementTask(schema, config=measurement_config)
    measCat = measurement.generateMeasCat(exposure, sourceCat, exposure.getWcs()) #Create measCat. This is where the measurements are stored

    ### Step 3: Run forced photometry ###
    measurement.run(measCat, exposure, sourceCat, exposure.getWcs())

    
    ### Step 4: Save results and return
    df_forced = measCat.asAstropy().to_pandas()
    df_forced['base_PsfFlux_SNR'] = df_forced['base_PsfFlux_instFlux'] / df_forced['base_PsfFlux_instFluxErr']
    if not return_full:    
        #This df contains a lot of information. Extract the important ones. We can also calculate the SNR:
        df_forced = df_forced[['base_PsfFlux_instFlux','base_PsfFlux_instFluxErr','base_PsfFlux_SNR','base_PsfFlux_area']]

    return df_forced









### =================== ###
### Combining the above ###
### =================== ###


def inject_subtract_photometry(science_exposure, injection_mag, sources, injection_locations, template_exposure = [], add_noise = False, plot = False, 
            sn_position = [], cutout_size = np.nan, plot_zoom = False, plot_zoom_size = 40, subtraction_config = None, band = 'g'):
    '''
    add_noise: add extra noise. Useful when debugging.

    cutout size is in arcsec
    
    TODO: should add try/except blocks here, sometimes the injection or subtraction doesn't work for too big
    sources. But then we're only injecting stars, so probably fine.


    TODO: Do parallelisation here too?
    '''

    if template_exposure == []:
        template_exposure = science_exposure.clone()

    df_results = pd.DataFrame()
    wcs = science_exposure.getWcs()


    if ~np.isnan(cutout_size) and sn_position != []:
        pixscale = science_exposure.getWcs().getPixelScale().asArcseconds()
        cutout_size_pix = int(cutout_size/pixscale)
        cutout_position = sn_position
    else:
        cutout_size_pix = int(np.min([science_exposure.width,science_exposure.height]))
        b = science_exposure.getBBox()
        cutout_position = (b.centerX, b.centerY)
        cutout_position = pixel_to_sky(cutout_position, wcs)
        
    science_cutout = cutout_exposure(science_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix, size_units = 'pixel')
    template_cutout = cutout_exposure(template_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix+20, size_units = 'pixel')
    
    
    for i in range(len(injection_locations)):
        # Inject SNe
        locs_sky = injection_locations[i]
        
        injection_catalog = create_injection_catalog(list(np.array(locs_sky)[:,0]),list(np.array(locs_sky)[:,1]),
                                                     ['Star']*len(locs_sky),[injection_mag]*len(locs_sky))



        injected_exposure, _ = inject_source(science_exposure, injection_catalog, band = band)



        if add_noise: # Only for debugging purposes
            # add noise
            injected_exposure = add_noise(injected_exposure, mu = 1, stdev = 0.05, method = 'multiplicative')
            injected_exposure = add_noise(injected_exposure, mu = 0, stdev = 1, method = 'additive')
            # add PSF
            #injected_exposure.setPsf(psfs[2])


        injected_cutout = cutout_exposure(injected_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix, size_units = 'pixel')


        # Subtract
        sources_cutout = filter_sources(sources, template_cutout) #Only keep the sources that are actually also in the injected calexps cutout
        sources_cutout = filter_sources(sources_cutout, injected_cutout) #Only keep the sources that are actually also in the injected calexps cutout
        diff_cutout = subtract_images(template_cutout, injected_cutout, sources_cutout, subtraction_config = subtraction_config)



    
        # Forced photometry
        results = forced_photometry(diff_cutout.difference, locs_sky)


        # Save results
        locs_pix = sky_to_pixel(locs_sky, wcs = science_exposure.getWcs())
        results['n_injection_iteration'] = i
        results['injection_x'] = np.array(locs_pix)[:,0]
        results['injection_y'] = np.array(locs_pix)[:,1]
        results['injection_ra'] = list(np.array(locs_sky)[:,0])
        results['injection_dec'] = list(np.array(locs_sky)[:,1])
        results['injection_id'] = [f'{i}_' + str(k) for k in list(range(len(results)))]
    
        df_results = pd.concat([df_results,results])




    if plot == True:
        if plot_zoom == False:
            b = science_exposure.getBBox()
            zoom_target = (b.centerX, b.centerY)
            zoom_target = pixel_to_sky(zoom_target, wcs = science_exposure.getWcs())
            plot_zoom_size = math.floor(science_exposure.width * science_exposure.getWcs().getPixelScale().asArcseconds())
            
        else:
            assert sn_position != [], "Need to specify a SN position if you want to zoom in."
            zoom_target = sn_position
        
        ncol = 3 # Uninjected, injected, diff
        nrow = 1

        fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*4,3*nrow))

        plot_image(science_cutout, ax = ax[0], title = 'Science exposure', zoom_target = zoom_target, zoom_size = plot_zoom_size)
        plot_image(injected_cutout, ax = ax[1], title = 'Injected science exposure', zoom_target = zoom_target, zoom_size = plot_zoom_size)
        plot_image(diff_cutout.difference, ax = ax[2], title = 'Difference', zoom_target = zoom_target, zoom_size = plot_zoom_size)
        

        plt.tight_layout()
        plt.show()


    return df_results




        




def sample_sn_mags(df, sn_mag, n_mag_step, buffer, mag_limits):
    '''
    Rather than just uniformly sampling a SN range, we want to spend more time in the region of the recovery curve where
    it transitions from 1 to 0. This function helps you do that.
    '''
    

    if len(df) == 0: #In first iteration, just sample around SN mag.
        sn_mags_inject = np.linspace(sn_mag - buffer, sn_mag + buffer,n_mag_step)
        if sn_mag not in sn_mags_inject: #Make sure the actual SN mag is always also being sampled
            sn_mags_inject = np.append(sn_mags_inject, sn_mag)
            
        
    else: # In future iterations, sample more directed
        df = df.sort_values(by = 'sn_mag').reset_index(drop=True)
        df = recovery_summary(df)
        
        # Find lower limit
        try:
            idx = np.where(df['detected'] == 1)[0][-1] #Highest one where it is fully detected
            mag_min = df['sn_mag'][idx]
        except:
            mag_min = mag_limits[0]

        # Find upper limit
        try:
            idx = np.where(df['detected'] == 0)[0][0] #Lowest one where it is not detected at all
            mag_max = df['sn_mag'][idx]
        except: #If the thresholds weren't found in the first iteration, just take max range.
            mag_max = mag_limits[1]



        # Create range
        sn_mags_inject = np.linspace(mag_min, mag_max ,n_mag_step+2)#+2 because by defitition, we will already have covered min and max. And so we will add less new injection sites than we think. We're filtering them (and any other duplicates) out in the next step
        sn_mags_inject = [i for i in sn_mags_inject if i not in np.array(df['sn_mag'])]

    return sn_mags_inject





class recovery_curve_config:
    '''
    Description parameters

    We try to be smart while sampling the recovery curve. We will have multiple rounds of sampling and slowly zoom in to the
    region where the curve goes from 1 to 0. The following parameters define this process.
    n_mag_steps: list of ints. Defines how many mag steps in each iteration. Better to start low, and go up. 
    sampling_buffer: float. During the first iteration, we create a range around the sn_mag. The buffer defines how far out on each side. 
    mag_limits: list of floats. If the transition region is not found in the first iteration, we simply set the limits of the search to this.

    Related to image smoothing:
    smooth_function: the function used to smooth
    smooth_filter_size: int. The size of the filter, in pixels

    Related to finding injection locations:
    injection_method: can be either 'random' or 'smooth'. When random, we just return random pixels in the inmage. When smooth, we first smooth the template image, and then try to find pixels that are similar to the target SN. Very highly recommended to use smooth.
    n_injections: int or list of ints. The amount of injections at each iteration. The precision on the detection fraction depends on this. E.g. if n_injections = 10, then we will only be precise with steps of 0.1 (i.e. 1/n_injections).
    max_inj_per_round: int. The maximum number of injections done per injection iteration. 
    p_threshold: int. We try to find injection sites that are similar to the target SN. Every injection site will be p_threshold percent within the value of the pixel at the SN location in the template image.
    psf_flux_threshold: float. Used when creating the injection iterations. We do not want to plce injection sites too close together. We look at the PSF of each injection location and find the distance at which 1 - psf_flux_threshold of the flux of the normalized PSF is contained and make sure to exclude other sites within that range.
    psf_sigma: In addition to the above, we also calculate the std of the PSF and add a psf_sigma multiple of that to this radius. 
    min_dist_across_iterations: even across injection iterations, we don't simply want to take the next pixel over. So whenever we find a suitable injection location, also mark all the other possible ones that are within X arcsec as unuseable.
    injection_n_attempts: Since the creation of the final injection locations from all the possible injection locations is a random process, we try a few times, and see if there is a suitable solution.
    max_dist: (float) the maximum distance, in arcsec, away from the SN that the injection locations can be. 
    inject_band: str. Band to inject in. Default 'g'. 
    
    Remaining parameters:
    subtraction_config: We use the AlardLuptonSubtractTask to do the subtraction. Note that you can also pass a `config` parameter to `subtract_images()`. This should be a `lsst.ip.diffim.subtractImages.AlardLuptonSubtractConfig`. This allows you to finetune the details of the subtraction. If you do not specify this parameter, we will use the default config. Strongly recommended to keep this to default, unless you're confident in what you want to change. 
    snr_threshold: float. What SNR threshold defines a detection. Default Rubin value is 5. 
    subtract_background: bool. Whether or not to automatically subtract the background.
    cutout_size: int. We recommend to input the full images, but we fill create a cutout ourselves for the majority of the tasks to speed things up. Units are arcsec. Default is NaN, but very highly recommended to use a value. It needs to be high enough to still include other sources for the subtraction. Around 300 seems  usually appropriate.
    sn_position_units: string. Whether the sn_position is in pixel units or sky units.
    plot: Whether to automatically output plots.
    expand_output: bool. If True, we expand the output to contain much more detail. You can recover the summarised dataframe by doing: `df_recovery = recovery_summary(df_expanded_output)`
    n_jobs: Used to specify how many parallel processes should be used to complete this. Should not exceed amount of available CPU cores. Can disable this functionality by doing n_jobs = 0. Parallelisation happens over inject_subtract_photometry() with different sn_mags. 
    '''

    
    def __init__(self):

        # SN sampling
        self.n_mag_steps = [4,4,4,8,20]
        self.sampling_buffer = 5.0
        self.mag_limits = [10,30]

        # Image smoothing
        self.injection_method = 'smooth'
        self.smooth_function = 'median'
        self.smooth_filter_size = 10
        
        # Finding injection locations
        self.n_injections = 10
        self.max_inj_per_round = 100
        self.p_threshold = 5
        self.psf_flux_threshold = 0.0001
        self.psf_sigma = 3
        self.min_dist_across_iterations = 1
        self.injection_n_attempts = 10
        self.max_dist = 50
        self.inject_band = 'g'

        #Subtraction
        self.subtraction_config = None
    
        # Varia
        self.snr_threshold = 5
        self.subtract_background = True
        self.cutout_size = np.nan
        self.sn_position_units = 'sky'
        self.plot = False
        self.expand_output = False
        self.n_jobs = 0


def recovery_curve(sn_mag, science_exposure, template_exposure, sources, sn_position, config = None):
    '''
    You have an LSST science exposure (calexp) and template exposure (coadd). There is a SN in the science exposure of magnitude sn_mag. It is embedded
    in a background galaxy. This function will help to decide whether the detection is real
    or not by creating a recovery curve.

    sn_mag: magnitude of the supernova.
    science_exposure: lsst.afw.image._exposure.ExposureF that contains the SN. Typically a calexp
    template_exposure: lsst.afw.image._exposure.ExposureF of the same field without the SN. Typically a coadd.
    sources: a SourceCatalog of all the sources in the science exposure. Used to do the subtraction.
    sn_position: the position of the SN, in pixel coordinates
    '''
    if config == None:
        config = recovery_curve_config()

    
    df_results_full = pd.DataFrame()
    
    n_iterations = len(config.n_mag_steps)


    assert config.injection_method in ['random', 'smooth'], "config.injection_method can be either 'random' or 'smooth'."
    assert config.sn_position_units in ['sky','pixel'], "config.sn_position_units can either be 'sky' or 'pixel'."



    if type(config.n_injections) == int:
        config.n_injections = [config.n_injections]*n_iterations

    if config.sn_position_units == 'pixel': #change it to sky position units
        sn_position = pixel_to_sky(sn_position, wcs = science_exposure.getWcs())




    # Create the cutouts
    wcs = science_exposure.getWcs()
    if ~np.isnan(config.cutout_size):
        pixscale = wcs.getPixelScale().asArcseconds()
        cutout_size_pix = int(config.cutout_size/pixscale)
    else:
        cutout_size_pix = int(np.min([science_exposure.width,science_exposure.height]))
        
    cutout_position = sn_position
    science_cutout = cutout_exposure(science_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix, size_units = 'pixel')
    template_cutout = cutout_exposure(template_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix+20, size_units = 'pixel')


    """
    # Create the cutouts
    wcs = science_exposure.getWcs()
    if ~np.isnan(cutout_size):
        pixscale = wcs.getPixelScale().asArcseconds()
        
        cutout_size_pix = int(cutout_size/pixscale)
        cutout_position = sn_position
        science_cutout = cutout_exposure(science_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix)
        template_cutout = cutout_exposure(template_exposure, cutout_position[0], cutout_position[1], size = cutout_size_pix+20)
    
    else:
        cutout_size_pix = int(np.min([science_exposure.width,science_exposure.height]))
        science_cutout = science_exposure.clone()
        template_cutout = template_exposure.clone()
    """


    
    # Create injection sites
    if config.injection_method == 'random':
        injection_locations = create_injection_locations(science_cutout, np.max(config.n_injections), sn_position = sn_position, 
                                                         method = 'random', psf_flux_threshold = config.psf_flux_threshold, 
                                                         psf_sigma= config.psf_sigma, max_per_round = config.max_inj_per_round,
                                                         min_dist_across_iterations = config.min_dist_across_iterations, 
                                                         n_attempts = config.n_attempts, max_dist = config.max_dist)

    if config.injection_method == 'smooth':
        injection_locations = create_injection_locations(science_cutout, np.max(config.n_injections), sn_position = sn_position, method = 'smooth', template_exposure = template_cutout, 
                                                        smooth_function = config.smooth_function, smooth_filter_size = config.smooth_filter_size,
                                                         p_threshold = config.p_threshold, psf_flux_threshold = config.psf_flux_threshold, 
                                                         psf_sigma = config.psf_sigma, max_per_round = config.max_inj_per_round,
                                                         min_dist_across_iterations = config.min_dist_across_iterations, 
                                                         n_attempts = config.injection_n_attempts, max_dist = config.max_dist)





    # Measure background
    df_background = measure_background(science_cutout, template_cutout, injection_locations, sources, subtraction_config = config.subtraction_config)
    df_background = df_background.rename(columns = {'base_PsfFlux_instFlux' : 'background_PsfFlux_instFlux', 'base_PsfFlux_instFluxErr' : 'background_PsfFlux_instFluxErr'})
    df_background['background_nonzero_flag'] = [1 if (df_background['background_PsfFlux_instFlux'][i] - df_background['background_PsfFlux_instFluxErr'][i]) > 0 else 0 for i in range(len(df_background))]

    

    for j in tqdm(range(n_iterations)): #Iterations here is iteration of SN magnitudes! 


        ### Step 1: Create injection SN range of this iteration
        sn_mags_inject = sample_sn_mags(df_results_full, sn_mag, config.n_mag_steps[j], config.sampling_buffer, config.mag_limits)
        if len(sn_mags_inject) == 0:  #We have already measured all these datapoints. Move on to next iteration. Happens in cases where more zoom isn't possible with current amount of datapoints
            continue


        
        ### Step 2: Loop over injection SN range of this iteration
        injection_locations_temp = lower_injection_locations(injection_locations, np.max(config.n_injections) - config.n_injections[j]) #If the amount of injection locations is not the same in every iteratino, remove a few here.


        df_results = pd.DataFrame()
        
        if config.n_jobs == 0: # If you want to disable the parallelisation
            for sn_mag_inject in sn_mags_inject:
                df_temp = inject_subtract_photometry(science_exposure, injection_mag = sn_mag_inject, 
                                  sources = sources, injection_locations = injection_locations_temp,
                                  template_exposure = template_exposure, cutout_size = config.cutout_size,
                                  sn_position = sn_position, subtraction_config = config.subtraction_config, 
                                  band = config.inject_band)
                
                df_temp['sn_mag'] = sn_mag_inject        
                df_results = pd.concat([df_results,df_temp])

        elif config.n_jobs >= 1: #Enable parallelisation
            with Pool(processes=config.n_jobs) as pool:

                inject_subtract_fixed = partial(inject_subtract_photometry, 
                                        sources = sources, injection_locations = injection_locations_temp,
                                        template_exposure = template_exposure, cutout_size = config.cutout_size,
                                        sn_position = sn_position, subtraction_config = config.subtraction_config,
                                        band = config.inject_band
                                        )

                args = [(science_exposure, sn_mag_inject) for sn_mag_inject in sn_mags_inject]
                
                temp = pool.starmap(inject_subtract_fixed, args)
        
            for k, sn_mag_inject in enumerate(sn_mags_inject):
                temp[k]['sn_mag'] = sn_mag_inject
                
            df_results = pd.concat(temp, ignore_index=True)



        
        ### Step 3: Save data
        df_results['sn_mag_iteration'] = j
        df_results['n_injections'] = config.n_injections[j]


        # Subtract background, if wanted
        df_results = df_results.merge(df_background, left_on = 'injection_id', right_on = 'injection_id')
        if config.subtract_background:
            df_results['base_PsfFlux_instFlux'] = df_results['base_PsfFlux_instFlux'] - df_results['background_PsfFlux_instFlux']
            df_results['base_PsfFlux_instFluxErr'] = np.sqrt(df_results['base_PsfFlux_instFluxErr']**2 + df_results['background_PsfFlux_instFluxErr']**2)
            df_results['base_PsfFlux_SNR'] = df_results['base_PsfFlux_instFlux'] / df_results['base_PsfFlux_instFluxErr']

        
        df_results['detected'] = [True if df_results.iloc[i]['base_PsfFlux_SNR'] > config.snr_threshold else False for i in range(len(df_results))]
        #df_results = df_results.reset_index(drop=True)
        #df_results = df_results.groupby('sn_mag').mean().reset_index().sort_values(by = 'sn_mag').reset_index(drop=True)

        # Combine iterations
        df_results_full = pd.concat([df_results_full, df_results])


    
    df_results_full = df_results_full.sort_values(by = 'sn_mag').reset_index(drop=True)
    df_summary = recovery_summary(df_results_full)
    
    if config.plot:
        plt.figure(figsize = (5,3))
        plt.plot(df_summary['sn_mag'], df_summary['detected'])
        plt.ylabel('Detection fraction')
        plt.xlabel('SNe magnitude')
        plt.axvline(sn_mag, c = 'k', ls = '--')
        plt.show()


    if config.expand_output:
        return df_results_full
    else: 
        return df_summary




def recovery_summary(df):
    df_summary = df.copy(deep=True)[['base_PsfFlux_instFlux', 'base_PsfFlux_instFluxErr', 'base_PsfFlux_SNR', 'sn_mag','detected','background_nonzero_flag']]
    df_summary = df_summary.groupby('sn_mag').mean().reset_index().sort_values(by = 'sn_mag').reset_index(drop=True)
    return df_summary





### ========== ###
### Background ###
### ========== ###

def measure_background(science, template, injection_locations, sources, subtraction_config = None):
    '''
    Background here means the difference image. Will subtract science and template, and then do forced photometry on 
    injection locations on the difference images. Should be consistent with 0. 
    '''

    df_results = pd.DataFrame()
    

    # Subtract
    sources = filter_sources(sources, template) #Only keep the sources that are actually also in the injected calexps cutout
    sources = filter_sources(sources, science) #Only keep the sources that are actually also in the injected calexps cutout
    diff = subtract_images(template, science, sources, subtraction_config = subtraction_config)



    for i in range(len(injection_locations)):
        locs_sky = injection_locations[i]

        # Forced photometry
        results = forced_photometry(diff.difference, locs_sky)

        results = results[['base_PsfFlux_instFlux','base_PsfFlux_instFluxErr']]
        results['injection_id'] = [f'{i}_' + str(k) for k in list(range(len(results)))]
    
        df_results = pd.concat([df_results,results])
        
    return df_results.reset_index(drop=True)




def estimate_sn_background(science, template, sources, sn_position, n_injections = np.inf,
                           sn_position_units = 'sky', cutout_size = np.nan, smooth_function = 'median', 
                           smooth_filter_size = 10, p_threshold = 2, max_dist = 10, psf_flux_threshold = 0.0001, 
                           psf_sigma = 3, radius_increment = 2, plot = False, subtraction_config = None):

    '''
    cutout_size is in arcsec
    '''

    
    assert sn_position_units in ['sky','pixel'], "sn_position_units can either be 'sky' or 'pixel'."


    if sn_position_units == 'pixel': #change it to sky position units
        sn_position = pixel_to_sky(sn_position, wcs = science.getWcs())


    # Create the cutouts
    wcs = science.getWcs()
    if ~np.isnan(cutout_size):
        pixscale = wcs.getPixelScale().asArcseconds()
        cutout_size_pix = int(cutout_size/pixscale)
    else:
        cutout_size_pix = int(np.min([science.width,science.height]))

    
    cutout_position = sn_position
    science_cutout = cutout_exposure(science, cutout_position[0], cutout_position[1], size = cutout_size_pix, size_units = 'pixel')
    template_cutout = cutout_exposure(template, cutout_position[0], cutout_position[1], size = cutout_size_pix+20, size_units = 'pixel')



    # Find all pixels that are within p_threshold percentile within the SN location
    injection_locations, _ = find_smoothened_injection_locations(template, sn_position, smooth_function = smooth_function, 
                                                               smooth_filter_size = smooth_filter_size, p_threshold = p_threshold, 
                                                               output_units = 'sky')
    
    # Remove pixels that are too close (or too far away) from SN. Might be some leaking. 
    min_dists = create_min_dists(science, injection_locations, psf_flux_threshold = psf_flux_threshold, psf_sigma = psf_sigma, radius_increment = radius_increment)
    sn_min_dist = create_min_dists(science, [sn_position], psf_flux_threshold = psf_flux_threshold, psf_sigma = psf_sigma, radius_increment = radius_increment)
    injection_locations, min_dists = filter_injection_locations(injection_locations,min_dists,[sn_position], sn_min_dist, remove_max_dists = [max_dist/3600])

    
    # Select a random subset
    if len(injection_locations) > n_injections:
        injection_locations = np.array(random.sample(list(injection_locations), n_injections))
    
    
    # Measure background at these locations
    df_background = measure_background(science, template, [injection_locations], sources, subtraction_config = subtraction_config)


    # Converty to njy
    njy = instflux_to_njy(df_background['base_PsfFlux_instFlux'], [science] * len(df_background))
    njy_err = instflux_to_njy(df_background['base_PsfFlux_instFluxErr'], [science] * len(df_background))
    df_background['base_PsfFlux_nJy'] = njy
    df_background['base_PsfFlux_nJyErr'] = njy_err

    df_background = df_background.drop(['base_PsfFlux_instFlux', 'base_PsfFlux_instFluxErr'], axis=1)


    df_background['injection_ra'] = injection_locations[:,0]
    df_background['injection_dec'] = injection_locations[:,1]

    if plot:

        plt.figure(figsize = (10,3))

        plt.subplot(1,2,1)
        plot_image(science, coords = injection_locations, zoom_target = sn_position)

        plt.subplot(1,2,2)
        plt.hist(df_background['base_PsfFlux_nJy'], bins = 20)
        plt.xticks(rotation = 45)
        plt.xlabel('Background PsfFlux [nJy]')
        plt.show()

    return df_background


def measure_snr_on_img(img, sn_position, zoom_size = 40, plot = False, stepsize = 1):

    ra, dec = sn_position
    
    cutout = cutout_exposure(img, ra, dec, size = zoom_size, size_units = 'sky')
    cutout_shape = cutout.height, cutout.width
    xmin, ymin = cutout.image.getX0(), cutout.image.getY0()
    xmax = xmin + cutout_shape[1]
    ymax = ymin + cutout_shape[0]
    
    xs = np.arange(xmin,xmax,stepsize)
    ys = np.arange(ymin,ymax,stepsize)
    
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()
    coords_pix = np.array([xs,ys]).T

    coords_sky = pixel_to_sky(coords_pix, wcs = img.getWcs()) 


    # Do the measurement
    df_forced = forced_photometry(cutout, coords_sky)
        
    #This df contains a lot of information. Extract the important ones. We can also calculate the SNR:
    df_forced = df_forced[['base_PsfFlux_instFlux','base_PsfFlux_instFluxErr','base_PsfFlux_area']]
    df_forced['base_PsfFlux_SNR'] = df_forced['base_PsfFlux_instFlux'] / df_forced['base_PsfFlux_instFluxErr']
    
    df_forced['x'] = xs
    df_forced['y'] = ys


    # Plot
    if plot:

        fig, ax = plt.subplots(1,2, figsize=(8, 3))
        plot_image(img, ax = ax[0],coords = [[ra,dec]], zoom_target = (ra,dec), zoom_size = zoom_size)

        plt.subplot(1,2,2)
        # Reshape the data into a grid
        x_values = df_forced['x'].unique()
        y_values = df_forced['y'].unique()
        z_grid = df_forced.pivot(index='y', columns='x', values='base_PsfFlux_SNR').values
        
        # Plot with pcolormesh
        plt.pcolormesh(x_values, y_values, z_grid, shading='auto', cmap='viridis', vmin = 0, vmax = 5)
        plt.xlim(xmin,xmax)
        plt.ylim(ymin, ymax)
        plt.colorbar(label = 'SNR')

        plt.tight_layout()
        plt.show()

    return df_forced
        






### =============== ###
### Find thresholds ###
### =============== ###




def piecewise(xs, x1, x2):
    '''
    y = 1 if x < x1. 
    y = 0 if x > x2. 
    and a line in between.
    '''

    ys = []
    m = -1 / (x2-x1)
    b = x2 / (x2-x1)

    for x in xs:
        if x < x1:
            ys.append(1)
        elif x > x2:
            ys.append(0)
        else:
            ys.append(m*x + b)

    return ys



def piecewise_solver(y,x1,x2):
    '''
    The nice thing about this is that you can solve it analytically. If y = 0 or 1, it is in the flat parts.
    In between, it is just a line. Solve y = m * x + b  <=> x = (y-b)/m and substitute.
    '''

    assert y >= 0 and y <= 1, "y must be between 0 and 1."

    return x2 - y * (x2-x1)





def find_thresholds(df_results, detection_fraction_thresholds = [0.5,0.8], p0 = [22,25], method = 'piecewise', sn_mag = np.nan, plot = False):
    '''
    For every item in detection_fraction_thresholds, find the lowest mag that has a higher detection fraction.
    '''

    assert method in ['empirical','piecewise']

    if method == 'empirical':
        d = {}
        
        for ft in detection_fraction_thresholds: 
            idx = np.where(df_results['detected'] >= ft)[0] #df is ordered on sn mag, take the index of the highest sn mag where this is true
            if len(idx) == 0:
                d[f'lim_{ft}'] = np.nan
            else:
                idx = idx[-1]
    
                lim_mag = df_results['sn_mag'][idx]
                d[f'lim_{ft}'] = lim_mag
    
        return d


    elif method == 'piecewise':
        xs_data = np.array(df_results['sn_mag'])
        ys_data = np.array(df_results['detected'])

        popt, _ = curve_fit(piecewise, xs_data, ys_data, p0=p0)

        d = {}
        for ft in detection_fraction_thresholds: 
            lim_mag = piecewise_solver(ft, *popt)
            d[f'lim_{ft}'] = lim_mag


        if plot == True:

            plt.figure(figsize = (6,4))
            xs = np.linspace(np.min(xs_data), np.max(xs_data), 10000)

            plt.plot(xs, piecewise(xs, *popt), c = 'k', label = 'model', zorder = -1, lw = 1)
            plt.scatter(xs_data,ys_data, label = 'data')

            if ~np.isnan(sn_mag):
                plt.axvline(sn_mag, c = 'k', ls = '--')
            

            for ft in detection_fraction_thresholds:
                plt.scatter(d[f'lim_{ft}'], ft, c = 'C1', marker = '+', zorder = 3, s = 100)
            plt.scatter([],[], c = 'C1', marker = '+', s = 100, label = 'Thresholds')
            
            plt.xlim(popt[0]-0.2,popt[1]+0.2)
            plt.xlabel('Magnitude')
            plt.ylabel('Detection fraction')
            plt.legend()
            plt.show()
            
        return d



### ================ ###
### Unit conversions ###
### ================ ###


def njy_to_mag(njy):
    '''
    Jansky is spectral flux density. 10**-26 watts m-2 Hz-1 or 10**-23 erg s-1 cm-2 Hz-1.
    magnitude is in AB mags

    More info: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648
    https://sites.astro.caltech.edu/~george/ay21/Fluxes%20and%20Magnitudes.pdf

    Can transform Jansky to AB mags by doing: S [uJy] = 10**((23.9 - AB)/2.5), but can also just use astropy. 
    Note that the zero-point magnitude here is 23.9, as it corresponds to AB mag system and flux in uJy in example. 
    For nJy, zero-point mag in AB system is 31.4 (as m = -2.5 log10(F) + m0 <=> m0 = m + 2.5 log10(F) => m0 = 0 + 2.5 log10(3631*10**9) = 31.4)

    This function assumes inputs to be in nanoJansky. Output in AB mag.
    '''

    single_target = False
    if isinstance(njy, (int, float, np.integer, np.floating)):
        njy = [njy]
        single_target = True

    
    mags = []
    for i in njy:
        mag = (i*u.nJy).to(u.ABmag).value
        mags.append(mag)

    if single_target:
        return mags[0]
    else:
        return mags



def mag_to_njy(mags):
    '''
    Jansky is spectral flux density. 10**-26 watts m-2 Hz-1 or 10**-23 erg s-1 cm-2 Hz-1.
    magnitude is in AB mags

    More info: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648
    https://sites.astro.caltech.edu/~george/ay21/Fluxes%20and%20Magnitudes.pdf

    Can transform Jansky to AB mags by doing: S [uJy] = 10**((23.9 - AB)/2.5), but can also just use astropy. 
    Note that the zero-point magnitude here is 23.9, as it corresponds to AB mag system and flux in uJy in example. 
    For nJy, zero-point mag in AB system is 31.4 (as m = -2.5 log10(F) + m0 <=> m0 = m + 2.5 log10(F) => m0 = 0 + 2.5 log10(3631*10**9) = 31.4)

    This function assumes inputs to be in nanoJansky. Output in AB mag.
    '''

    single_target = False
    if isinstance(mags, (int, float, np.integer, np.floating)):
        mags = [mags]
        single_target = True

    
    njy = []
    for i in mags:
        temp = (i*u.ABmag).to(u.nJy).value
        njy.append(temp)

    if single_target:
        return njy[0]
    else:
        return njy
        





def instflux_to_njy(instflux, exp):
    '''
    More info: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648

    Going from instflux (counts) to njy, using the calibration of each exposure. Done with:

    instflux [counts] * calibration = flux [njy]

    This is done with the photoCalib object, which is attached to every exposure. PhotoCalib has methods to transform to njy or mags.

    Note: You can use this funtion to convert instflux errors to njy errors.
    '''

    single_target = False
    if isinstance(instflux, (int, float, np.integer, np.floating)):
        instflux = [instflux]
        exp = [exp]
        single_target = True


    njy = []
    for i in range(len(instflux)):
        photocalib = exp[i].getPhotoCalib()
        temp = photocalib.instFluxToNanojansky(instflux[i])
        njy.append(temp)


    if single_target:
        return njy[0]
    else:
        return njy


def instflux_to_mag(instflux, exp):
    '''
    More info: https://community.lsst.org/t/photocalib-has-replaced-calib-welcoming-our-nanojansky-overlords/3648

    Going from instflux (counts) to njy, using the calibration of each exposure. Done with:

    instflux [counts] * calibration = flux [njy]

    This is done with the photoCalib object, which is attached to every exposure. PhotoCalib has methods to transform to njy or mags.

    Note: You cannot use this function to convert instflux errors to mag errors! Only values. 
    '''


    single_target = False
    if isinstance(instflux, (int, float, np.integer, np.floating)):
        instflux = [instflux]
        exp = [exp]
        single_target = True


    mags = []
    for i in range(len(instflux)):
        photocalib = exp[i].getPhotoCalib()
        temp = photocalib.instFluxToMagnitude(instflux[i])
        mags.append(temp)


    if single_target:
        return mags[0]
    else:
        return mags    




def njyerr_to_magerr(njyerr, njy):
    '''
    Flux-to-Magnitude Formula:

    $m=−2.5log⁡10(F)$
    
    Differentiating the Magnitude Formula with respect to flux:
        
    $frac{dm}{dF} = -2.5 frac{1}{ln(10) F}$ 
    
    Apply error propagation:
    
    $sigma_{m} = |frac{dm}{dF}| sigma_{F}$
    
    $Rightarrow sigma_{m} = frac{2.5}{ln(10)} frac{1}{F} sigma_{F}$ 

    Shouldn't depend on unit of flux though, as long as flux and its error are in the same units. This is also how scisql_nanojanskyToAbMagSigma() works.
    '''



    single_target = False
    if isinstance(njyerr, (int, float, np.integer, np.floating)):
        njyerr = [njyerr]
        njy = [njy]
        single_target = True

    magerrs = []
    for i in range(len(njyerr)):
        magerr = 2.5 / np.log(10) * 1/njy[i] * njyerr[i]
        magerrs.append(magerr)



    if single_target:
        return magerrs[0]
    else:
        return magerrs





def mag_to_absmag(lst, d = None, z = None, cosmo = None):
    '''
    Convert between apparent and absolute magnitude.
    '''

    # Check if it is a single target or a list of targets. If single, wrap it in a list.
    single_target = False
    if type(lst) in [int,float, np.floating, np.float64,np.int64]:
        lst = [lst]
        d = [d]
        z = [z]
        single_target = True


    if cosmo == None:
        if len(d) == 1:
            d = [d[0]]*len(lst)
    if cosmo != None:
        if len(z) == 1:
            z = [z[0]]*len(lst)


    if cosmo != None:
        d = cosmo.luminosity_distance(np.array(z)).to('pc').value


            

    absmags = []
    for i,m in enumerate(lst): # Don't have to do the loop...
        M = m - 5 * np.log10(d[i]) + 5
        absmags.append(M)
    

    if single_target:
        return absmags[0]
    else:
        return np.array(absmags)

    

def absmag_to_mag(lst, d = None, z = None, cosmo = None):
    '''
    Convert between apparent and absolute magnitude.
    '''

    # Check if it is a single target or a list of targets. If single, wrap it in a list.
    single_target = False
    if type(lst) in [int,float, np.floating, np.float64,np.int64]:
        lst = [lst]
        d = [d]
        z = [z]
        single_target = True


    if cosmo == None:
        if len(d) == 1:
            d = [d[0]]*len(lst)
    if cosmo != None:
        if len(z) == 1:
            z = [z[0]]*len(lst)



    if cosmo != None:
        d = cosmo.luminosity_distance(z).to('pc').value
            

    mags = []
    for i,M in enumerate(lst): # Don't have to do the loop...
        m = M + 5 * np.log10(d[i]) - 5
        mags.append(m)
    

    if single_target:
        return mags[0]
    else:
        return np.array(mags)
    

    






### ===== ###
### Varia ###
### ===== ###




def rotate_exposure(exp, n_degrees):
    """
    Copied from the DP02_14 tutorial notebook on 26 Feb 2025
    
    Rotate an exposure by nDegrees clockwise.

    Parameters
    ----------
    exp : `lsst.afw.image.exposure.Exposure`
        The exposure to rotate
    n_degrees : `float`
        Number of degrees clockwise to rotate by

    Returns
    -------
    rotated_exp : `lsst.afw.image.exposure.Exposure`
        A copy of the input exposure, rotated by nDegrees
    """
    n_degrees = n_degrees % 360

    wcs = exp.getWcs()

    warper = afwMath.Warper('lanczos4')

    affine_rot_transform = geom.AffineTransform.makeRotation(n_degrees*geom.degrees)
    transform_p2top2 = afwGeom.makeTransform(affine_rot_transform)
    rotated_wcs = afwGeom.makeModifiedWcs(transform_p2top2, wcs, False)

    rotated_exp = warper.warpExposure(rotated_wcs, exp)
    return rotated_exp



class stopwatch:
    '''
    Make a stopwatch class that makes timing stuff easier.
    '''
    def __init__(self):
        # Start when initialised
        self.start_time = time.time()
        self.lap_time = time.time()
        
        
    def start(self):
        # Start again
        self.start_time = time.time()
        self.lap_time = time.time()

    def reset(self):
        # Wrapper around start
        self.start()
        
    
    def read(self, output = True, ):
        # Read how much time has passed since start
    
        time_passed = time.time() - self.start_time 
        
        if output:
            self.output(time_passed)
            
            
    def lap(self, output = True):
        laptime_passed = time.time() - self.lap_time 
        if output:
            self.output(laptime_passed)
        
        self.lap_time = time.time()
    
            
    def output(self, val, precision = 2):
        # Print out time
        '''
        TODO: can output in secs, mins our hours
        '''
        print(f'{np.round(val, precision)} seconds have passed.')
