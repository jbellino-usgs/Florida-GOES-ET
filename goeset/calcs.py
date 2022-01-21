import numpy as np


# Fixed variables for ET calculations
P_atm = 101.3
gsc = 0.082
stefan_MJ = 4.903e-9
stefan_W = 5.67e-8
e_surface = 0.97

# Locally calibrated Sellers' parameters
a1 = 0.575
a2 = 0.054
alpha = 1.26


def avg_arrays(arrays):
    """
    Take the average of multiple arrays.

    arrays : list of numpy ndarrays

    """
    return np.ma.sum(arrays, axis=0) / len(arrays)


def calc_saturation_vapor_pressure(t):
    """
    Calculate the saturation vapor pressure at air temperature T.
    Equation 11 of Allen (1998). Temperature is in Celsius.

    Parameters
    ----------
    t : numpy ndarray
        Air temperature, in degrees Celsius

    """
    return 0.6108 * np.exp((17.27 * t) / (t + 237.3))


def calc_actual_vapor_pressure(es_tmin, es_tmax, hmin, hmax):
    """
    Calculate actual vapor pressure from saturation vapor pressure
    and relative humidity. Equation 17 of Allen (1998).

    Parameters
    ----------
    es_tmin : numpy ndarray
        Saturation vapor pressure at minimum air temperature.
    es_tmax : numpy ndarray
        Saturation vapor pressure at maximum air temperature.
    hmin : numpy ndarray
        Minimum relative humidity, in percent
    hmax : numpy ndarray
        Maximum relative humidity, in percent

    """
    return ((es_tmax * hmin / 100.) + (es_tmin * hmax / 100.)) / 2.


def calc_saturation_vapor_pressure_curve_delta(t):
    """
    Calculate the slope of the saturation vapor pressure curve at air temperature T.
    Equation 13 of Allen (1998).

    Parameters
    ----------
    t : numpy ndarray
        Air temperature, in degrees Celsius

    """
    return 4098 * (0.6108 * np.exp((17.27 * t) / (t + 237.3))) / ((t + 237.3) ** 2.)


def calc_inverse_earth_sun_dist(j, ndays):
    """
    Calculate the inverse Earth-sun distance for Julian day(s) J.
    Equation 23 of Allen (1998).

    Parameters
    ----------
    j : numpy ndarray
        Julian day
    ndays : int
        Number of days in the year

    """
    return 1. + 0.033 * np.cos(2. * np.pi * (j - .5) / ndays)


def calc_declination(j):
    """
    Calculate the declination of the sun for Julian day(s) J. Note the original Fortran code used [2pi/366] * J.
    Equation 24 of Allen (1998).

    Parameters
    ----------
    j : numpy ndarray
        Julian day

    """

    return 0.409 * np.sin(2. * np.pi * (j - .5) / 365. - 1.39)


def calc_rnl(tmin, tmax, ea, fcd):
    """
    Equation 39 of Allen (1998). Net outgoing longwave radiation.

    Parameters
    ----------
    tmin : numpy ndarray of floats
        Minimum daily air temperature, in degrees Celsius
    tmax : numpy ndarray of floats
        Maximum daily air temperature, in degrees Celsius
    ea : numpy ndarray
        Actual vapor pressure
    *fcd : numpy ndarray
        Total solar radiation divided by clear-sky solar radiation

    * Not in equation 39 of Allen (1998), logic was present in Fortran script provided by W.B. Shoemaker
    """
    return stefan_MJ * (((tmax + 273.15) ** 4. + (tmin + 273.15) ** 4.) / 2.) * (0.34 - 0.14 * np.sqrt(ea)) * fcd


def calc_ws(rad_lat, declination):
    """
    Solar time angle at mid day.

    Parameters
    ----------
    rad_lat : numpy ndarray
        Latitude in radians.
    declination : numpy ndarray
        Solar declination.

    """
    xx = 1.0 - (((np.tan(rad_lat)) ** 2.) * ((np.tan(declination)) ** 2.))
    ws = np.pi / 2. - np.arctan((-np.tan(rad_lat)) * (np.tan(declination)) / (xx ** 0.5))
    return ws


def calc_ra(dr, ws, rad_lat, declination):
    """
    Compute extraterrestrial radiation
    Variation on equation 28 of Allen (1998)
    Note: Allen uses 12 hours, but Jacobs paper uses 24.

    Parameters
    ----------
    dr : numpy ndarray
        Inverse relative Earth-sun distance.
    ws : numpy ndarray
        Solar time angle at mid day.
    rad_lat : numpy ndarray
        Latitude in radians.
    declination : numpy ndarray
        Solar declination.

    """
    ra = (24. * 60. / np.pi) * gsc * dr * (ws * (np.sin(rad_lat)) *
                                           (np.sin(declination)) + (np.cos(rad_lat)) * (np.cos(declination)) * (
                                               np.sin(ws)))
    return ra


def calc_rns(rs, albedo):
    """
    Total incoming shortwave radiation.

    Parameters
    ----------
    rs : numpy ndarray
        Total incoming shortwave solar radiation, in MegaJoules per square meter per day
    albedo : numpy ndarray
        Shortwave blue-sky albedo, unitless

    """
    return (1. - albedo) * rs


def calc_rlu(t):
    """
    Upwelling longwave radiation.

    Parameters
    ----------
    t : numpy ndarray
        Air temperature, in degrees Celsius

    """
    return e_surface * stefan_W * (t + 273.15) ** 4.


def calc_rso(ra, frac=0.75):
    """
    Clear-sky solar radiation.

    Parameters
    ----------
    ra : numpy ndarray
        Extraterrestrial solar radiation.
    frac : float <= 1
        Fraction of extraterrestrial solar radiation that
        reaches earth on clear-sky days.

    """
    s = 'Please enter a fractional value less than or equal to 1.0 and ' \
        'greater than or equal to 0.0.'
    assert 0 <= frac <= 1, s
    return ra * frac


def calc_fcd(rso):
    """
    # IF((RS(ss)/Rso) .GT. 1.0)THEN
    #     fcd = 1.0
    # ELSE IF((RS(ss)/Rso) .LT. 0.3)THEN
    #     fcd = 0.05
    # ELSE
    #     fcd = 1.35 * (RS(ss)/Rso) - 0.35
    # ENDIF

    Parameters
    ----------
    rso : numpy ndarray
        Clear-sky solar radiation.

    """
    fcd = rso.copy()
    fcd[rso > 1] = 1.
    fcd[rso < 0.3] = 0.05
    fcd[((rso >= 0.3) & (rso <= 1))] = (1.35 * fcd[((rso >= 0.3) & (rso <= 1))]) - .35
    return fcd


def calc_rldc(ea, t):
    """
    Clear sky downwelling longwave radiation.

    Parameters
    ----------
    ea : numpy ndarray
        Actual vapor pressure
    t : numpy ndarray
        Air temperature, in degrees Celsius

    """
    return (a1 + a2 * np.sqrt(10. * ea)) * stefan_W * (t + 273.15) ** 4.


def calc_clf(rs, rso):
    """
    Crawford and Duchon (1999) cloud fraction.

    Parameters
    ----------
    rs : numpy ndarray
        Total incoming shortwave solar radiation, in MegaJoules per square meter per day
    rso : numpy ndarray
        Clear-sky solar radiation.

    """
    clf = 1.0 - (rs / rso)

    # From Crawford and Duchon (1999):
    # Calculated values of clf less than zero were adjusted back to
    # zero so as to be physically realistic.
    clf[clf < 0] = 0

    return clf


def calc_rld(rldc, clf, t):
    """
    Cloudy sky downwelling longwave radiation.

    Parameters
    ----------
    rldc : numpy ndarray of floats
        Clear sky downwelling longwave radiation.
    clf : numpy ndarray of floats
        Crawford and Duchon (1999) cloud fraction.
    t : numpy ndarray of floats
        Air temperature, in degrees Celsius.

    """
    s = 'Please enter a cloud-fraction value less than or equal to 1.0 and ' \
        'greater than or equal to 0.0.'
    assert np.all((0 <= clf) & (clf <= 1), where=(~np.isnan(clf))), s
    return rldc * (1. - clf) + clf * stefan_W * (t + 273.15) ** 4.


def calc_ern(rns, rld, rlu):
    """
    Energy Balance (ERn)
    Conversion from W/m2 to MJ/m2

    Parameters
    ----------
    rns : numpy ndarray of floats
        Net shortwave radiation.
    rld : numpy ndarray of floats
        Cloudy sky downwelling longwave radiation.
    rlu : numpy ndarray of floats
        Upwelling longwave radiation.

    """
    return rns + e_surface * rld * 0.0864 - rlu * 0.0864


def calc_eto(dates, tmin, tmax,  hmin,  hmax, ws2m, rs, lat, albedo=0.23):
    """
    Allen, R.G., Pereira, L.S., Paes, D., and Smith, M., 1998, Crop
    evapotranspiration - Guidelines for computing crop water requirements -
    FAO Irrigation and drainage paper 56.

    Parameters
    ----------
    dates : numpy ndarray of datetime.date objects
        Dates for which potential evapotranspiration are to be computed
    tmin : numpy ndarray of floats
        Daily minimum air temperature, in degrees Celsius
    tmax : numpy ndarray of floats
        Daily maximum air temperature, in degrees Celsius
    hmin : numpy ndarray of floats
        Daily minimum percent relative humidity
    hmax : numpy ndarray of floats
        Daily maximum percent relative humidity
    rs : numpy ndarray of floats
        Total incoming shortwave solar radiation, in MegaJoules per square meter per day
    ws2m : numpy ndarray of floats
        Daily average wind speed at 2-meter height, in meters per second
    lat : numpy ndarray of floats
        Latitude, in degrees
    albedo : float
        Albedo, unitless (Albedo for RET is 0.23.)

    """
    # Number of days in the year.
    ndays = len(dates)

    # Julian days
    jday = np.array(list(range(1, ndays + 1)))

    # Atmospheric pressure is 101.3.  Can adjust to temperature, if desired.
    # Use optional equations embedded in DO loop.
    #      P_atm = 101.3*((293.0-(0.0065*elevation))/293.0)**5.26
    #      P_atm = 101.3*((Temp/293.0)**5.26
    gamma = 0.665 * 10.0 ** (-3.0) * P_atm

    # Compute average temperature and humidity
    tavg = avg_arrays([tmax, tmin])

    # Compute saturation vapor pressure (eq. 12, Allen [1998])
    # Not equivalent to saturation vapor pressure of average temperature.
    es_tmin = calc_saturation_vapor_pressure(tmin)
    es_tmax = calc_saturation_vapor_pressure(tmax)
    es = avg_arrays([es_tmin, es_tmax])

    # Calculate actual vapor pressure
    ea = calc_actual_vapor_pressure(es_tmin, es_tmax, hmin, hmax)

    # Calculate the slope of the saturation vapor pressure curve
    delta = calc_saturation_vapor_pressure_curve_delta(tavg)

    # Latitude in radians.
    rad_lat = np.array([(np.pi / 180.) * lat] * ndays)

    # Calculate inverse relative Earth-sun distance
    dr = np.ones_like(rad_lat)
    for j in jday:
        dr[j - 1] *= calc_inverse_earth_sun_dist(j, ndays)

    # Calculate solar declination
    declination = np.ones_like(rad_lat)
    for j in jday:
        declination[j - 1] *= calc_declination(j)

    # Calculate solar time angle at mid day
    ws = calc_ws(rad_lat, declination)

    # Compute extraterrestrial radiation
    ra = calc_ra(dr, ws, rad_lat, declination)

    # Clear-sky solar radiation where 75 percent of extraterrestrial
    # radiation reaches earth on clear-sky days.
    rso = calc_rso(ra, frac=0.75)

    # Net shortwave radiation.
    rns = calc_rns(rs, albedo)

    # Compute total solar radiation divided by clear-sky solar radiation.
    rs_rso = rs / rso

    # Create new array "fcd", equal to total solar radiation
    # divided by clear-sky solar radiation.
    fcd = rs_rso.copy()

    # IF((RS(ss)/Rso) .GT. 1.0)THEN
    #     fcd = 1.0
    # ELSE IF((RS(ss)/Rso) .LT. 0.3)THEN
    #     fcd = 0.05
    # ELSE
    #     fcd = 1.35 * (RS(ss)/Rso) - 0.35
    # ENDIF

    fcd[rs_rso > 1] = 1.
    fcd[rs_rso < 0.3] = 0.05
    fcd[((rs_rso >= 0.3) & (rs_rso <= 1))] = (1.35 * fcd[((rs_rso >= 0.3) & (rs_rso <= 1))]) - .35

    # Calculate net outgoing longwave radiation
    rnl = calc_rnl(tmin, tmax, ea, fcd)

    # Equation 40 of Allen(1998). Net radiation as the difference between
    # the incoming net shortwave radiation and the outgoing net longwave
    # radiation
    rn = rns - rnl

    # Equation 6 of Allen(1998). Reference evapotranspiration based on Food and Agriculural
    # Organization of the United Nations, Penman-Monteith equation. Temperature converted
    # to Kelvin (note Allen used 273.16).
    # Assume soil heat flux density is zero (term in Rn-0.0).
    g = 0.
    eto = (0.408 * delta * (rn - g) + gamma * (900.0 * ws2m * (es - ea)) / (tavg + 273.15)) / (
                delta + gamma * (1. + 0.34 * ws2m))
    return eto


def calc_pet(dates, tmin, tmax, hmin, hmax, rs, lat, albedo):
    """
    Priestly-Taylor (1972) potential evapotranspiration.

    Parameters
    ----------
    dates : numpy ndarray of datetime.date objects
        Dates for which potential evapotranspiration are to be computed
    tmin : numpy ndarray of floats
        Daily minimum air temperature, in degrees Celsius
    tmax : numpy ndarray of floats
        Daily maximum air temperature, in degrees Celsius
    hmin : numpy ndarray of floats
        Daily minimum percent relative humidity
    hmax : numpy ndarray of floats
        Daily maximum percent relative humidity
    rs : numpy ndarray of floats
        Total incoming shortwave solar radiation, in MegaJoules per square meter per day
    lat : numpy ndarray of floats
        Latitude, in degrees
    albedo : numpy ndarray of floats
        Shortwave blue-sky albedo, unitless

    """

    # Number of days in the year.
    ndays = len(dates)

    # Julian days
    jday = np.array(list(range(1, ndays + 1)))

    # Compute average temperature and humidity
    tavg = avg_arrays([tmax, tmin])
    havg = avg_arrays([hmax, hmin])

    # Atmospheric pressure is 101.3.  Can adjust to temperture, if desired.
    # Use optional equations embedded in DO loop.
    #      P_atm = 101.3*((293.0-(0.0065*elevation))/293.0)**5.26
    #      P_atm = 101.3*((Temp/293.0)**5.26
    gamma = 0.665 * 10.0 ** (-3.0) * P_atm

    # Calculate the slope of the saturation vapor pressure curve
    delta = calc_saturation_vapor_pressure_curve_delta(tavg)

    # Net shortwave radiation
    rns = calc_rns(rs, albedo)

    # Compute saturation vapor pressure
    es_tmax = calc_saturation_vapor_pressure(tmax)
    es_tmin = calc_saturation_vapor_pressure(tmin)

    # Calculate actual vapor pressure
    ea = calc_actual_vapor_pressure(es_tmin, es_tmax, hmin, hmax)

    # Calculate clear sky downwelling longwave radiation
    rldc = calc_rldc(ea, tavg)

    # Latitude in radians.
    rad_lat = np.array([(np.pi / 180.) * lat] * ndays)

    # Calculate inverse relative Earth-sun distance
    dr = np.ones_like(rad_lat)
    for j in jday:
        dr[j - 1] *= calc_inverse_earth_sun_dist(j, ndays)

    # Calculate solar declination
    declination = np.ones_like(rad_lat)
    for j in jday:
        declination[j - 1] *= calc_declination(j)

    # Calculate solar time angle at mid day
    ws = calc_ws(rad_lat, declination)

    # Compute extraterrestrial radiation
    ra = calc_ra(dr, ws, rad_lat, declination)

    # Clear-sky solar radiation where 75 percent of extraterrestrial
    # radiation reaches earth on clear-sky days.
    rso = calc_rso(ra, frac=0.75)

    # Crawford and Duchon (1999) cloud fraction.
    clf = calc_clf(rs, rso)

    # Cloudy sky downwelling longwave radiation.
    rld = calc_rld(rldc, clf, tavg)

    # Upwelling longwave radiation
    rlu = calc_rlu(tavg)

    # Energy Balance
    ern = calc_ern(rns, rld, rlu)

    # Convert from  𝑊/𝑚2  to  𝑚𝑚/𝑑  by dividing by the density and latent heat of vaporization of water:
    # ((2.501−0.002361𝑇)×1000)×1000
    # Assume soil heat flux density is zero (term in Rn-0.0).
    g = 0.
    return (alpha * (delta / (delta + gamma)) * (ern - g)) / ((2.501 - 0.002361 * tavg) * 1000.) * 1000.
