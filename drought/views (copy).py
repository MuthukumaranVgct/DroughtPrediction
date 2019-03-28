from django.shortcuts import render
import xlrd
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import scipy.special
import scipy.stats
import os,math
import numba
from math import *
# Create your views here.
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09

from bs4 import BeautifulSoup
import requests
import urllib.parse as u
import requests
import json





def getDistrict():
    send_url = 'https://ipinfo.io'
    r = requests.get(send_url)
    print('r',r)
    print('json',json.loads(r.text))
    j = json.loads(r.text)['loc']
    lat = j[:j.find(',')]
    lon = j[j.find(',')+1:]
    print(lat,lon)

    #lat= 11.004556 #float(input('Enter your latitude'))
    #lon= 76.961632 #float(input('Enter your longitude'))
    URL = 'https://www.latlong.net/Show-Latitude-Longitude.html'
    s = requests.Session()

    def fetch(url, data=None):
        if data is None:
            return s.get(url).content
        else:
            return s.post(url, data=data).content

    soup = BeautifulSoup(fetch(URL),'lxml')
    form = soup.findAll('form')

    fields = form[0].findAll('input')
 
    formdata = dict( (field.get('name'), field.get('value')) for field in fields)     
    formdata['latitude'] = lat 
    formdata['longitude'] = lon
    formdata['btn'] = 'Submit'


    r = s.post(URL, data=formdata)


    tempSoup = BeautifulSoup(r.text,'lxml')

    tablediv = tempSoup.find('div', {'id':'address'})

    address=tablediv.find(text=True)
    print(address)
    if address.find('Tamil')==-1:
        print('Your place should be in Tamil Nadu')
        return 'None'
    left=address[:address.find('Tamil')-2]
    district=left
    if left.rfind(',')!=-1:
        district=left[left.rfind(',')+2:]
    print(district)
    return district


precMap={'ARIYALUR':'data (1).xls',
'CHENNAI':'data (2).xls',
'COIMBATORE':'data (3).xls',
'CUDDALORE':'data (4).xls',
'DHARMAPURI':'data (5).xls',
'DINDIGUL':'data (6).xls',
'ERODE':'data (7).xls',
'KANCHEEPURAM':'data (8).xls',
'KANNIYAKUMARI':'data (30).xls',
'KARUR':'data (9).xls',
'MADURAI':'data (11).xls',
'NAGAPATTINAM':'data (12).xls',
'NAMAKKAL':'data (13).xls',
'PERAMBALUR':'data (14).xls',
'PUDUKKOTTAI':'data (15).xls',
'RAMANATHAPURAM':'data (16).xls',
'SALEM':'data (17).xls',
'SIVAGANGA':'data (18).xls',
'THANJAVUR':'data (19).xls',
'THENI':'data (20).xls',
'NILGIRIS':'data (21).xls',
'THIRUVALLUR':'data (22).xls',
'THIRUVARUR':'data (23).xls',
'THOOTHUKKUDI':'data (24).xls',
'TIRUCHIRAPALLI':'data (25).xls',
'TIRUNELVELI':'data (26).xls',
'TIRUVANNAMALAI':'data (27).xls',
'VELLORE':'data (28).xls',
'VILLUPURAM':'data (29).xls',
'KALLAKURICHI':'data (29).xls',
'VIRUDHUNAGAR':'data (30).xls',
}

tempMap={'ARIYALUR':'Ariyalur.xls',
'CHENNAI':'Chennai.xls',
'COIMBATORE':'Coimbatore.xls',
'CUDDALORE':'Cuddalore.xls',
'DHARMAPURI':'Dharmapuri.xls',
'DINDIGUL':'Dindigul.xls',
'ERODE':'Erode.xls',
'KANCHEEPURAM':'Kanchipuram.xls',
'KANNIYAKUMARI':'data (30).xls',
'KARUR':'data (10).xls',
'MADURAI':'data (11).xls',
'NAGAPATTINAM':'data (12).xls',
'NAMAKKAL':'data (13).xls',
'PERAMBALUR':'data (14).xls',
'PUDUKKOTTAI':'data (15).xls',
'RAMANATHAPURAM':'data (16).xls',
'SALEM':'data (17).xls',
'SIVAGANGA':'data (18).xls',
'THANJAVUR':'data (19).xls',
'THENI':'data (20).xls',
'NILGIRIS':'data (21).xls',
'THIRUVALLUR':'data (22).xls',
'THIRUVARUR':'data (23).xls',
'THOOTHUKKUDI':'data (24).xls',
'TIRUCHIRAPALLI':'data (25).xls',
'TIRUNELVELI':'data (26).xls',
'TIRUVANNAMALAI':'data (27).xls',
'VELLORE':'data (28).xls',
'VILLUPURAM':'data (29).xls',
'KALLAKURICHI':'data (29).xls',
'VIRUDHUNAGAR':'data (30).xls',
}

PETMap={'ARIYALUR':'data (1).xls',
'CHENNAI':'data (2).xls',
'COIMBATORE':'data (3).xls',
'CUDDALORE':'data (4).xls',
'DHARMAPURI':'data (5).xls',
'DINDIGUL':'data (6).xls',
'ERODE':'data (7).xls',
'KANCHEEPURAM':'data (8).xls',
'KANNIYAKUMARI':'data (30).xls',
'KARUR':'data (9).xls',
'MADURAI':'data (11).xls',
'NAGAPATTINAM':'data (12).xls',
'NAMAKKAL':'data (13).xls',
'PERAMBALUR':'data (14).xls',
'PUDUKKOTTAI':'data (15).xls',
'RAMANATHAPURAM':'data (16).xls',
'SALEM':'data (17).xls',
'SIVAGANGA':'data (18).xls',
'THANJAVUR':'data (19).xls',
'THENI':'data (20).xls',
'NILGIRIS':'data (21).xls',
'THIRUVALLUR':'data (22).xls',
'THIRUVARUR':'data (23).xls',
'THOOTHUKKUDI':'data (24).xls',
'TIRUCHIRAPALLI':'data (25).xls',
'TIRUNELVELI':'data (26).xls',
'TIRUVANNAMALAI':'data (27).xls',
'VELLORE':'data (28).xls',
'VILLUPURAM':'data (29).xls',
'KALLAKURICHI':'data (29).xls',
'VIRUDHUNAGAR':'data (30).xls',
}

lat={
'ARIYALUR':'',
'CHENNAI':'',
'COIMBATORE':'',
'CUDDALORE':'data (4).xls',
'DHARMAPURI':'data (5).xls',
'DINDIGUL':'data (6).xls',
'ERODE':'data (7).xls',
'KANCHEEPURAM':'data (8).xls',
'KANNIYAKUMARI':'data (30).xls',
'KARUR':'data (9).xls',
'MADURAI':'data (11).xls',
'NAGAPATTINAM':'data (12).xls',
'NAMAKKAL':'data (13).xls',
'PERAMBALUR':'data (14).xls',
'PUDUKKOTTAI':'data (15).xls',
'RAMANATHAPURAM':'data (16).xls',
'SALEM':'data (17).xls',
'SIVAGANGA':'data (18).xls',
'THANJAVUR':'data (19).xls',
'THENI':'data (20).xls',
'NILGIRIS':'data (21).xls',
'THIRUVALLUR':'data (22).xls',
'THIRUVARUR':'data (23).xls',
'THOOTHUKKUDI':'data (24).xls',
'TIRUCHIRAPALLI':'data (25).xls',
'TIRUNELVELI':'data (26).xls',
'TIRUVANNAMALAI':'data (27).xls',
'VELLORE':'data (28).xls',
'VILLUPURAM':'data (29).xls',
'KALLAKURICHI':'data (29).xls',
'VIRUDHUNAGAR':'data (30).xls',

}

def reshape_to_2d(values,
                  second_axis_length):
    '''
    :param values: an 1-D numpy.ndarray of values
    :param second_axis_length: 
    :return: the original values reshaped to 2-D, with shape (int(original length / second axis length), second axis length)
    :rtype: 2-D numpy.ndarray of floats
    '''
    
    # if we've been passed a 2-D array with valid shape then let it pass through
    shape = values.shape
    if len(shape) == 2:
        if shape[1] == second_axis_length:
            # data is already in the shape we want, return it unaltered
            return values
        else:
            message = 'Values array has an invalid shape (2-D but second dimension not 12): {}'.format(shape)
            _logger.error(message)
            raise ValueError(message)
    
    # otherwise make sure that we've been passed in a flat (1-D) array of values    
    elif len(shape) != 1:
        message = 'Values array has an invalid shape (not 1-D or 2-D): {0}'.format(shape)
        _logger.error(message)
        raise ValueError(message)

    # pad the end of the original array in order to have an ordinal increment, if necessary
    final_year_months = shape[0] % second_axis_length
    if final_year_months > 0:
        pad_months = second_axis_length - final_year_months
        pad_values = np.full((pad_months,), np.NaN)
        values = np.append(values, pad_values)
        
    # we should have an ordinal number of years now (ordinally divisible by second_axis_length)
    increments = int(values.shape[0] / second_axis_length)
    
    # reshape from (months) to (years, 12) in order to have one year of months per row
    return np.reshape(values, (increments, second_axis_length))



def sum_to_scale(values,
                 scale):
    '''
    Compute a sliding sums array using 1-D convolution. The initial (scale - 1) elements 
    of the result array will be padded with np.NaN values. Missing values are not ignored, i.e. if a np.NaN
    (missing) value is part of the group of values to be summed then the sum will be np.NaN
    
    For example if the first array is [3, 4, 6, 2, 1, 3, 5, 8, 5] and the number of values to sum is 3 then the resulting
    array will be [np.NaN, np.NaN, 13, 12, 9, 6, 9, 16, 18].
    
    More generally:
    
    Y = f(X, n)
    
    Y[i] == np.NaN, where i < n
    Y[i] == sum(X[i - n + 1:i + 1]), where i >= n - 1 and X[i - n + 1:i + 1] contains no NaN values
    Y[i] == np.NaN, where i >= n - 1 and X[i - n + 1:i + 1] contains one or more NaN values
         
    :param values: the array of values over which we'll compute sliding sums
    :param scale: the number of values for which each sliding summation will encompass, for example if this value
                  is 3 then the first two elements of the output array will contain the pad value and the third 
                  element of the output array will contain the sum of the first three elements, and so on 
    :return: an array of sliding sums, equal in length to the input values array, left padded with NaN values  
    '''
    
    # don't bother if the number of values to sum is 1 (will result in duplicate array)
    if scale == 1:
        return values
    
    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(scale), mode='valid')
    
    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.NaN]*(scale - 1), sliding_sums))


def calculateSPI(district,scale):
    datasetName='DroughtPrediction/DatasetPrecipitation/'+precMap[district.upper()]
    df=pd.read_csv(datasetName,skiprows=51,delim_whitespace=True,names=['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    print(df)
    data_start_year=1950
    calibration_year_initial=1950
    calibration_year_final=2002
    feature_cols=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    precip1=df[feature_cols]
    precip=[]
    for row in precip1.iterrows():
        index,data=row
        precip+=data.tolist()
    precip=np.array(precip)
    print('=======================================',precip)
    scaled_precip=sum_to_scale(precip,scale)
    scaled_precips = reshape_to_2d(scaled_precip, 12)
    # fit the scaled values to a gamma distribution and transform to corresponding normalized sigmas 
    transformed_fitted_values_gamma = transform_fitted_gamma(scaled_precips, 
                                                                   data_start_year,
                                                                   calibration_year_initial,
                                                                   calibration_year_final,
                                                                   'monthly')
    # fit the scaled values to a Pearson Type III distribution and transform to corresponding normalized sigmas 
    transformed_fitted_values_pearson = transform_fitted_pearson(scaled_precips, 
                                                                     data_start_year,
                                                                     calibration_year_initial,
                                                                     calibration_year_final,
                                                                     'monthly')
        
    # clip values to within the valid range, reshape the array back to 1-D
    spi_gamma = np.clip(transformed_fitted_values_gamma, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()
    spi_pearson = np.clip(transformed_fitted_values_pearson, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()
    
    # return the original size array 
    #print(spi_gamma)
    count=1
    data=[]
    for i,j in zip(spi_gamma,precip):
        if count==13:
            count=1
        if str(i)!='nan':
            data.append([i,j,count])
        count+=1
    data=pd.DataFrame(data,columns=['SPI','Precip','time'])
    feature_cols = ['Precip','time']

    # use the list to select a subset of the original DataFrame
    X = data[feature_cols]
    feature_cols = ['SPI']
    y = data[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    #print(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    # instantiate
    linreg = LinearRegression()

    # fit the model to the training data (learn the coefficients)
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print('Predicted')
    print(type(y_test))
    print('Actual')
    '''for i in y_pred:
        print(i)'''
    print('RMSE Error')
    acc=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('AccuracyGamSPI',acc)
    dic={}
    dic['SPIGamPred']=y_pred[-1]
    dic['SPIGamAct']=list(y_test['SPI'])[-1]
    dic['accuracyGamma']=(1-(acc/4.0))*100
    #print(spi_pearson)
    count=1
    data=[]
    for i,j in zip(spi_pearson,precip):
        if count==13:
            count=1
        if str(i)!='nan':
            data.append([i,j,count])
        count+=1
    data=pd.DataFrame(data,columns=['SPI','Precip','time'])
    feature_cols = ['Precip','time']

    # use the list to select a subset of the original DataFrame
    X = data[feature_cols]
    feature_cols = ['SPI']
    y = data[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    #print(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    # instantiate
    linreg = LinearRegression()

    # fit the model to the training data (learn the coefficients)
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print('Predicted')
    #print(y_test)
    print('Actual')
    '''for i in y_pred:
        print(i)'''
    print('RMSE Error')
    acc=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('AccuracyPearSPI',acc)
    dic['SPIPearPred']=y_pred[-1]
    dic['SPIPearAct']=list(y_test['SPI'])[-1]
    dic['accuracyPearson']=(1-(acc/4.0))*100
    dic['scale']=scale
    return dic

def calculateSPEI(district,scale):
    datasetName='DroughtPrediction/DatasetPrecipitation/'+precMap[district.upper()]    
    df=pd.read_csv(datasetName,skiprows=51,delim_whitespace=True,names=['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    #print(df)
    data_start_year=1950
    calibration_year_initial=1950
    calibration_year_final=2002
    feature_cols=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    precip1=df[feature_cols]
    precip=[]
    for row in precip1.iterrows():
        index,data=row
        precip+=data.tolist()
    precip=np.array(precip)
    datasetName='DroughtPrediction/DatasetPET/'+PETMap[district.upper()]    
    df=pd.read_csv(datasetName,skiprows=51,delim_whitespace=True,names=['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    #print(df)
    feature_cols=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    pet1=df[feature_cols]
    pet=[]
    for row in pet1.iterrows():
        index,data=row
        pet+=data.tolist()
    pet=np.array(pet)
    p_minus_pet = (precip.flatten() - pet.flatten()) + 1000.0
    scaled_precPET=sum_to_scale(p_minus_pet,scale)
        
    # fit the scaled values to a gamma distribution and transform to corresponding normalized sigmas 
    transformed_fitted_values_gamma = transform_fitted_gamma(scaled_precPET, 
                                                                   data_start_year,
                                                                   calibration_year_initial,
                                                                   calibration_year_final,
                                                                   'monthly')

    # fit the scaled values to a Pearson Type III distribution and transform to corresponding normalized sigmas 
    transformed_fitted_values_pearson = transform_fitted_pearson(scaled_precPET, 
                                                                     data_start_year,
                                                                     calibration_year_initial,
                                                                     calibration_year_final,
                                                                     'monthly')
        
    # clip values to within the valid range, reshape the array back to 1-D
    spei_gamma = np.clip(transformed_fitted_values_gamma, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()
    spei_pearson = np.clip(transformed_fitted_values_pearson, _FITTED_INDEX_VALID_MIN, _FITTED_INDEX_VALID_MAX).flatten()

    # return the original size array 
    #print(spei_gamma)
    count=0
    data=[]
    for i,j,k in zip(spei_gamma,precip,pet):
        if count==13:
            count=1
        if str(i)!='nan':
            data.append([i,j,k,count])
        count+=1
    data=pd.DataFrame(data,columns=['spei','Precip','PET','time'])
    feature_cols=['Precip','PET','time']
    X = data[feature_cols]
    feature_cols = ['spei']
    y = data[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    #print(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    # instantiate
    linreg = LinearRegression()

    # fit the model to the training data (learn the coefficients)
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print('Predicted')
    #print(y_test)
    print('Actual')
    '''for i in y_pred:
        print(i)'''
    print('RMSE Error')
    acc=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('AccuracyGamSPEI',acc)
    dic={}
    dic['SPEIGamPred']=y_pred[-1]
    dic['SPEIGamAct']=list(y_test['spei'])[-1]
    dic['accuracyGamma']=(1-(acc/4.0))*100


    #print(spei_pearson)
    count=0
    data=[]
    for i,j,k in zip(spei_pearson,precip,pet):
        if count==13:
            count=1
        if str(i)!='nan':
            data.append([i,j,k,count])
        count+=1
    data=pd.DataFrame(data,columns=['spei','Precip','PET','time'])
    feature_cols=['Precip','PET','time']
    X = data[feature_cols]
    feature_cols = ['spei']
    y = data[feature_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    print(y_test)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # instantiate
    linreg = LinearRegression()

    # fit the model to the training data (learn the coefficients)
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    print('Predicted')
    print(y_test)
    print('Actual')
    for i in y_pred:
        print(i)
    print('RMSE Error')
    acc=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print('AccuracyPearSPEI',acc)
    
    dic['SPEIPearPred']=y_pred[-1]
    dic['SPEIPearAct']=list(y_test['spei'])[-1]
    dic['accuracyPearson']=(1-(acc/4.0))*100
    dic['scale']=scale
    return dic

def index(request):
    district=getDistrict()
    districts=['Chennai','Cuddalore','Kancheepuram','Thiruvallur','Thiruvannamalai','Vellore','Villuppuram','Dharmapuri','Krishnagiri','Namakkal','Salem','Erode','Thiruppur','Coimbatore','Nilgiris','Madurai','Theni','Dindigul','Sivagangai','Virudhunagar','Ramanathapuram','Thirunelveli','Thoothukudi','Kanyakumari','Pudhukkottai','Ariyalur','Nagapattinam','Perambalur','Thanjavur','Thiruchirappalli','Karur','Thiruvarur']

    print('Inside Index')
    if request.method=='POST':
        print('Request Got')
        district=request.POST['district'].strip()
        scale=int(request.POST['scale'])
        dicSPI=calculateSPI(district,scale)
        dicSPEI=calculateSPEI(district,scale)
        print(dicSPI['accuracyPearson'],dicSPI['accuracyGamma'],dicSPEI['accuracyPearson'],dicSPEI['accuracyGamma'])
        minAccur=max(dicSPI['accuracyPearson'],dicSPI['accuracyGamma'],dicSPEI['accuracyPearson'],dicSPEI['accuracyGamma'])
        if minAccur==dicSPEI['accuracyPearson']:
            SPEI=(dicSPEI['SPEIPearPred']+dicSPEI['SPEIPearAct'])/2
            if SPEI<-2:
                status='Extremely drought'
            elif SPEI<-1.50:
                status='Severe Drought'
            elif SPEI<-1.00:
                status='Moderately Drought'
            elif SPEI<0:
                status='Near Normal Drought'
            elif SPEI>2.00:
                status='Extremely Wet'
            elif SPEI>1.5:
                status='Very Wet'
            elif SPEI>1.0:
                status='Moderately Wet'
            else:
                status='Normal Condition'
            won='SPEI Pearson'
        elif minAccur==dicSPEI['accuracyGamma']:
            SPEI=(dicSPEI['SPEIGamPred']+dicSPEI['SPEIGamAct'])/2
            if SPEI<-2:
                status='Extremely drought'
            elif SPEI<-1.50:
                status='Severe Drought'
            elif SPEI<-1.00:
                status='Moderately Drought'
            elif SPEI<0:
                status='Near Normal Drought'
            elif SPEI>2.00:
                status='Extremely Wet'
            elif SPEI>1.5:
                status='Very Wet'
            elif SPEI>1.0:
                status='Moderately Wet'
            else:
                status='Normal Condition'
            won='SPEI Gamma'
        elif minAccur==dicSPI['accuracyPearson']:
            SPI=(dicSPI['SPIPearPred']+dicSPI['SPIPearAct'])/2
            if SPI<-2:
                status='Extremely drought'
            elif SPI<-1.50:
                status='Severe Drought'
            elif SPI<-1.00:
                status='Moderately Drought'
            elif SPI<0:
                status='Near Normal Drought'
            elif SPI>2.00:
                status='Extremely Wet'
            elif SPI>1.5:
                status='Very Wet'
            elif SPI>1.0:
                status='Moderately Wet'
            else:
                status='Normal Condition'
            won='SPI Pearson'
        else:
            SPI=(dicSPI['SPIGamPred']+dicSPI['SPIGamAct'])/2
            if SPI<-2:
                status='Extremely drought'
            elif SPI<-1.50:
                status='Severe Drought'
            elif SPI<-1.00:
                status='Moderately Drought'
            elif SPI<0:
                status='Near Normal Drought'
            elif SPI>2.00:
                status='Extremely Wet'
            elif SPI>1.5:
                status='Very Wet'
            elif SPI>1.0:
                status='Moderately Wet'
            else:
                status='Normal Condition'
            won='SPI Gamma'
        return render(request,'index.html',{'status':status,'accuracy':minAccur.round(2),'scale':scale,'places':districts,'place':district,'won':won})    
    if district!='None':
        return render(request,'index.html',{'places':districts,'place':district})
    else:
        return render(request,'index.html',{'places':districts})

