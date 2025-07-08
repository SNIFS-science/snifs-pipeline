Notes file for quick_cube.py


1. Start with input arc and frame.
  a. if the input is a continuum file, we dont expect a raster image, otherwise we do (see make_cubes.py:27)
2. Preprocess if needed. Boo, skip this part.
3. check_and_embed_raster
  a. checks for files under 30MB in size
  b. calls embed_raster with ccdsec '[974:1075,1:4096]'
  c. embeds the raster into a numpy array of detector size (using DETSIZE) - see embed_raster.py:90
  d. Apparently "There's a bug in R-raster: there's a *physical* offset of 1px between raster and full frames"
  e. updates DATASEC to please DS9 (line 122)
  f. adds a zero variance for some reason
  g. note that the embedding here is actually talking about embedding the raster extraction into a full frame array,
     not into another file. We're going to change that.
4. checks for mask. TODO: follow up with Greg about if we need this - quick_cube does not ever pass in a mask
5. instead, find the mask from the after_date, and hopefully I can just use that.
  a. TODO: ask Greg for the snifsmask files
  b. TODO: ask Greg for the refarc files
6. does fit_background.py, passing in the inframe only
  a. yses 3 sigma threshold and 1.3 smoothing
  b. checks for BKGNDSUB flag
  c. reads RDNOISE1, RDNOISE2, and RDNOISE from the header
  d. skips dark subtraction, as its already been done
  e. runs fitBackground, I believe with BkgIndexTable set to None
  f. now at fit_background.py:240, run a median filter on the signal, replace inf var signals with shitty imputation
  g. median filter the variance
  h. box filter in the dispersion direction # TODO: col/row flipping again, this should be the y-axis (4096 axis) right
  i. minimum filter in cross dispersion direction
  j. now use this filtered sig and var to determine "background" pixels,
     aka original pixels within 3 filtered sig of the filtered signal
  k. select at most 100k of these good signals (line 284) and create SmoothBivariateSpline (line 354) to map xy to signal
  l. subtract this interpolation from the data. dont change the variance
7. eugh  more keyword patching (quick_cub.py:413), sets arc fclass to 4 and frame fclass to 18
8. and now we pass off to actually extract_spec2.c, whic his 3k lines long jesus christ lord have mercy
  a. start at main, line 2265.
  b. Note: LENS is false (no_debug=-1)