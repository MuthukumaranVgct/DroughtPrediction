from django.shortcuts import render
import xlrd
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import scipy.special
from sklearn.tree import export_graphviz

from .calculations import *

from sklearn.ensemble import RandomForestRegressor
from math import *
from bs4 import BeautifulSoup
import requests
import urllib.parse as u
import requests
import json



# Create your views here.
_FITTED_INDEX_VALID_MIN = -3.09
_FITTED_INDEX_VALID_MAX = 3.09







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
'THIRUVANNAMALAI':'data (27).xls',
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
'THIRUVANNAMALAI':'data (27).xls',
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
'THIRUVANNAMALAI':'data (27).xls',
'TIRUVANNAMALAI':'data (27).xls',
'VELLORE':'data (28).xls',
'VILLUPURAM':'data (29).xls',
'KALLAKURICHI':'data (29).xls',
'VIRUDHUNAGAR':'data (30).xls',
}






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
    #linreg = LinearRegression()
    if scale!=1:
        rf= LinearRegression()
    else:
        rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

    # fit the model to the training data (learn the coefficients)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
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
    if scale!=1:
        rf= LinearRegression()
    else:
        rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

    # fit the model to the training data (learn the coefficients)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
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
    feature_cols=['Precip','time']
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
    if scale!=1:
        rf= LinearRegression()
    else:
        rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

    # fit the model to the training data (learn the coefficients)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
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
    feature_cols=['Precip','time']
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
    if scale!=1:
        rf= LinearRegression()
    else:
        rf = RandomForestRegressor(n_estimators = 500, random_state = 42)

    # fit the model to the training data (learn the coefficients)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
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

