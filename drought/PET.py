import calendar
import logging
import math
import numba
import numpy as np

#from climate_indices 
import utils

#-----------------------------------------------------------------------------------------------------------------------
# set up a basic, global _logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------

# days of each calendar month, for non-leap and leap years
_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

#-----------------------------------------------------------------------------------------------------------------------
# compute angle values used within the _sunset_hour_angle() function defined below

# valid range for latitude, in radians
_LATITUDE_RADIANS_MIN = np.deg2rad(-90.0)
_LATITUDE_RADIANS_MAX = np.deg2rad(90.0)

# valid range for solar declination angle, in radians
# Goswami (2015), p.40
_SOLAR_DECLINATION_RADIANS_MIN = np.deg2rad(-23.45)
_SOLAR_DECLINATION_RADIANS_MAX = np.deg2rad(23.45)

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _sunset_hour_angle(latitude_radians,
                       solar_declination_radians):
    '''
    Calculate sunset hour angle (*Ws*) from latitude and solar declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param latitude_radians: latitude in radians
    :param solar_declination_radians: angle of solar declination in radians
    :return: sunset hour angle in radians
    :rtype: float
    '''
    
    # validate the latitude argument
    if not _LATITUDE_RADIANS_MIN <= latitude_radians <= _LATITUDE_RADIANS_MAX:
        raise ValueError('latitude outside valid range [{0!r} to {1!r}]: {2!r}'
                         .format(_LATITUDE_RADIANS_MIN, _LATITUDE_RADIANS_MAX, latitude_radians))

    # validate the solar declination angle argument, which can vary between -23.45 and +23.45 degrees
    # see Goswami (2015) p.40, and http://www.itacanet.org/the-sun-as-a-source-of-energy/part-1-solar-astronomy/
    if not _SOLAR_DECLINATION_RADIANS_MIN <= solar_declination_radians <= _SOLAR_DECLINATION_RADIANS_MAX:
        raise ValueError('solar declination angle outside valid range [{0!r} to {1!r}]: {2!r}'
                         .format(_SOLAR_DECLINATION_RADIANS_MIN, _SOLAR_DECLINATION_RADIANS_MAX, solar_declination_radians))

    # calculate the cosine of the sunset hour angle (*Ws* in FAO 25) from latitude and solar declination
    cos_sunset_hour_angle = -math.tan(latitude_radians) * math.tan(solar_declination_radians)
    
    # If the sunset hour angle is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If the sunset hour angle is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    return math.acos(min(max(cos_sunset_hour_angle, -1.0), 1.0))

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _solar_declination(day_of_year):
    '''
    Calculate the angle of solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: day of year integer between 1 and 365 (or 366, in the case of a leap year)
    :return: solar declination [radians]
    :rtype: float
    :raise ValueError: if the day of year value is not within the range [1-366] 
    '''
    if not 1 <= day_of_year <= 366:
        raise ValueError('Day of the year must be in the range [1-366]: {0!r}'.format(day_of_year))

    return 0.409 * math.sin(((2.0 * math.pi / 365.0) * day_of_year - 1.39))

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _daylight_hours(sunset_hour_angle_radians):
    '''
    Calculate daylight hours from a sunset hour angle.

    Based on FAO equation 34 in Allen et al (1998).

    :param sunset_hour_angle_radians: sunset hour angle, in radians
    :return: number of daylight hours corresponding to the sunset hour angle
    :rtype: float
    :raise ValueError: if the sunset hour angle is not within valid range
    '''
    
    # validate the sunset hour angle argument, which has a valid range of 0 to pi radians (180 degrees), inclusive
    # see http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    if not 0.0 <= sunset_hour_angle_radians <= math.pi:
        raise ValueError('sunset hour angle outside valid range [{0!r} to {1!r}]: {2!r}'
                         .format(0.0, math.pi, sunset_hour_angle_radians))
    
    # calculate daylight hours from the sunset hour angle
    return (24.0 / math.pi) * sunset_hour_angle_radians

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def _monthly_mean_daylight_hours(latitude_radians, 
                                 leap=False):
    '''
    :param latitude_radians: latitude in radians
    :param leap: whether or not values should be computed specific to leap years
    :return: the mean daily daylight hours for each calendar month of a year
    :rtype: numpy.ndarray of floats, 1-D with shape: (12,)
    '''

    # get the array of days for each month based on whether or not we're in a leap year
    if not leap:
        month_days = _MONTH_DAYS_NONLEAP
    else:
        month_days = _MONTH_DAYS_LEAP
        
    # allocate an array of daylight hours for each of the 12 months of the year
    monthly_mean_dlh = np.zeros((12,))
    
    # keep a count of the day of the year
    day_of_year = 1
    
    # loop over each calendar month to calculate the daylight hours for the month
    for i, days_in_month in enumerate(month_days):
        cumulative_daylight_hours = 0.0   # cumulative daylight hours for the month
        for _ in range(1, days_in_month + 1):
            daily_solar_declination = _solar_declination(day_of_year)
            daily_sunset_hour_angle = _sunset_hour_angle(latitude_radians, daily_solar_declination)
            cumulative_daylight_hours += _daylight_hours(daily_sunset_hour_angle)
            day_of_year += 1
        
        # calculate the mean daylight hours of the month
        monthly_mean_dlh[i] = cumulative_daylight_hours / days_in_month
        
    return monthly_mean_dlh

latitude_degrees=11.1401
data_start_year=1901

monthly_temps_celsius=np.array([26.404,27.091,27.69,29.636,30.443,30.301,29.416,28.944,28.745,27.873,26.102,24.818,
25.318,25.292,27.879,30.147,30.917,30.502,29.415,29.132,28.372,27.074,26.089,25.502,
25.716,26.903,28.179,30.149,30.36,30.217,29.33,28.546,28.187,27.787,25.914,24.603,
24.815,25.304,27.464,30.149,30.171,29.712,28.928,28.63,28.957,27.874,26.4,24.591,
24.719,26.505,28.776,29.449,30.346,30.501,30.514,29.13,29.156,27.674,26.301,25.019])
original_length = monthly_temps_celsius.size

    # validate the input data array
monthly_temps_celsius = utils.reshape_to_2d(monthly_temps_celsius, 12)
print('Input(temperature per month)')
print('  Jan    Feb    March    April    May    June   July     August    Sep     Oct     Nov    Dec')
for i in monthly_temps_celsius:
    for j in i:
        print(j,end='  ')
    print('\n')
print('Latitude:',latitude_degrees)
print('Years:1901 to 1905')

    # at this point we assume that our dataset array has shape (years, 12) where 
    # each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)
    
    # convert the latitude from degrees to radians
latitude_radians = math.radians(latitude_degrees)
    
    # adjust negative temperature values to zero, since negative values aren't allowed (no evaporation below freezing)
    #TODO this sometimes throws a RuntimeWarning for invalid value, perhaps as a result of a NaN,
    # somehow use masking and/or NaN precheck to eliminate the cause of this warning
monthly_temps_celsius[monthly_temps_celsius < 0] = 0.0
    
    # mean the monthly temperature values over the month axis, giving us 12 monthly means for the period of record
mean_monthly_temps = np.nanmean(monthly_temps_celsius, axis=0)    
    
    # calculate the heat index (I)
I = np.sum(np.power(mean_monthly_temps / 5.0, 1.514))

    # calculate the a coefficient
a = (6.75e-07 * I ** 3) - (7.71e-05 * I ** 2) + (1.792e-02 * I) + 0.49239

    # get mean daylight hours for both normal and leap years 
mean_daylight_hours_nonleap = np.array(_monthly_mean_daylight_hours(latitude_radians, False))
mean_daylight_hours_leap = np.array(_monthly_mean_daylight_hours(latitude_radians, True))
    
    # allocate the PET array we'll fill
pet = np.full(monthly_temps_celsius.shape, np.NaN)
for year in range(monthly_temps_celsius.shape[0]):
        
    if calendar.isleap(data_start_year + year):
        month_days = _MONTH_DAYS_LEAP
        mean_daylight_hours = mean_daylight_hours_leap
    else:
        month_days = _MONTH_DAYS_NONLEAP
        mean_daylight_hours = mean_daylight_hours_nonleap

        # calculate the Thornthwaite equation
    pet[year, :] = 16 * (mean_daylight_hours / 12.0) * (month_days / 30.0) * ((10.0 * monthly_temps_celsius[year, :] / I) ** a)
    
    # reshape the dataset from (years, 12) into (months), i.e. convert from 2-D to 1-D, and truncate to the original length
print('Output(PET in mm per month)\n')
print('  Jan      Feb      March     April    May    June    July     August      Sep     Oct     Nov      Dec')
for i in pet:
    for j in i:
        print(j.round(3),end='  ')
    print('\n')
pet.reshape(-1)[0:original_length]