import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import cross_val_score
from joblib import dump, load

#-------Import Data Models---------#
filename_Model_AppAmps='201019_ECModel_AppAmps.pkl'
filename_Model_EffAmps='081019_ECModel_EffAmps.pkl'
model_AppAmps=load(filename_Model_AppAmps) #Quadratic Ridge model
model_EffAmps=load(filename_Model_EffAmps) #Quadratic equation model

#-------- Creat Dataframe with Setpoints-------------~#
def updatedf(Area_SPs,df):
    df_msk = df.copy()
    df.loc[:,:]=0
    n_features=df.columns.size
    n_samples=df.index.size
    # set all values less than 5 to 0
    cond1 = df_msk > 5
    df_msk = df_msk.where(cond1,0)
    # set all values greater than 0 to 1
    cond2 = df_msk < 1
    df_msk = df_msk.where(cond2,1)

    for i in range(0,4):
        df.iloc[:,i]=Area_SPs[2] #IGBT 9-12 = Area 3

    for i in range(4,8):
        df.iloc[:,i]=Area_SPs[3] #IGBT 13-16 = Area 4
    
    for i in range(8,12):
        df.iloc[:,i]=Area_SPs[4] #IGBT 17-20 = Area 5
    
    for i in range(12,16):
        df.iloc[:,i]=Area_SPs[5] #IGBT 21-24 = Area 6

    for i in range(16,20):
        df.iloc[:,i]=Area_SPs[6] #IGBT 25-28 = Area 7

    for i in range(20,24):
        df.iloc[:,i]=Area_SPs[7] #IGBT 29-32 = Area 8
    
    for i in range(24,26):
        df.iloc[:,i]=Area_SPs[8] #IGBT 33-34 = Area 9

    for i in range(26,28):
        df.iloc[:,i]=Area_SPs[9] #IGBT 35-36 = Area 10

    for i in range(28,30):
        df.iloc[:,i]=Area_SPs[10] #IGBT 37-38 = Area 11

    for i in range(30,32):
        df.iloc[:,i]=Area_SPs[11] #IGBT 39-40 = Area 12

    for i in range(32,34):
        df.iloc[:,i]=Area_SPs[12] #IGBT 41-42 = Area 13

    for i in range(34,36):
        df.iloc[:,i]=Area_SPs[13] #IGBT 43-44 = Area 14

    df=df*df_msk
    return df

#-------- Run Model on Setpoint dataframe-----------#
def RunModel(df):
    df['Coulombs'] = 0.0
    x=[0]*df.index.size
    App_Amps=pd.Series(x)
    for i in range(df.index.size): #calculate applied voltages & coulombs one line at a time
        df['Coulombs']=App_Amps.mul(2)
        df['Coulombs']=df['Coulombs'].fillna(0)
        df['Coulombs']=df['Coulombs'].cumsum()
        xtest=df.iloc[:(i+1),:].values #read features to a numpy array
        xtest.reshape((i+1),37) #reshape array (1 x n_features)
        y=model_AppAmps.predict(xtest) #using the model predict app for all IGBTs
        df_y=pd.DataFrame(data=y)
        #df_y[df_y < 0]=0
        App_Amps=df_y.sum(axis=1)

    df['App_Amps']=App_Amps

    dfx=df.loc[:,'Coulombs':'App_Amps']
    dfx=dfx.reindex(columns=['App_Amps','Coulombs'])
    xtest=dfx.values
    xtest.reshape(159,2)
    y=model_EffAmps.predict(xtest)
    cols=['Front_Amps','Roof_Amps','RHWing_Amps','LHWing_Amps','Rear_Amps']
    df_out=pd.DataFrame(data=y,columns=cols)
    df_out['Body_Amps']=dfx['App_Amps']
                            
    df_out['Front_Coulombs'] = df_out['Front_Amps'].cumsum()
    df_out['Roof_Coulombs'] = df_out['Roof_Amps'].cumsum()
    df_out['RHWing_Coulombs'] = df_out['RHWing_Amps'].cumsum()
    df_out['LHWing_Coulombs'] = df_out['LHWing_Amps'].cumsum()
    df_out['Rear_Coulombs'] = df_out['Rear_Amps'].cumsum()
    df_out['Body_Coulombs']=dfx['Coulombs']
    return df_out



    

#-------Get baseline data----------#
df_BaseLine=pd.read_csv('Data/BaseLineData.csv')

df_BaseLine['Front_Coulombs'] = df_BaseLine['Front_Amps'].cumsum()
df_BaseLine['Roof_Coulombs'] = df_BaseLine['Roof_Amps'].cumsum()
df_BaseLine['RHWing_Coulombs'] = df_BaseLine['RHWing_Amps'].cumsum()
df_BaseLine['LHWing_Coulombs'] = df_BaseLine['LHWing_Amps'].cumsum()
df_BaseLine['Rear_Coulombs'] = df_BaseLine['Rear_Amps'].cumsum()
df_BaseLine['Front_Coulombs'] = df_BaseLine['Front_Amps'].cumsum()
df_BaseLine['Body_Coulombs'] = df_BaseLine['Body_Amps'].cumsum()

BaseLine_coulombs = [df_BaseLine['Front_Amps'].mul(2).sum(),
                     df_BaseLine['Roof_Amps'].mul(2).sum(),
                     df_BaseLine['RHWing_Amps'].mul(2).sum(),
                     df_BaseLine['LHWing_Amps'].mul(2).sum(),
                     df_BaseLine['Rear_Amps'].mul(2).sum(),
                     df_BaseLine['Body_Amps'].mul(2).sum()]

#-------Make the base blank dataframe and the masking df-----#
dframe=pd.read_csv('Data/System_Validation_Features.csv')
dfbase=dframe.loc[:,'09V':'Coulombs']
results_df=pd.DataFrame()
df_in=pd.DataFrame()
results_available=None
show_baseline=st.sidebar.checkbox('Show Base Line')
show_resultsdf=st.sidebar.checkbox('Show Results DataFrame')
show_spdf=st.sidebar.checkbox('Show Setpoints DataFrame')

#-------Move Setpoints to the dataframe to be processed-------#
ASPs=[0]*14
ASPs[0]=st.sidebar.slider('Area 1',min_value=0, max_value=400, value=0)
ASPs[1]=st.sidebar.slider('Area 2',min_value=0, max_value=400, value=0)
ASPs[2]=st.sidebar.slider('Area 3',min_value=0, max_value=400, value=240)
ASPs[3]=st.sidebar.slider('Area 4',min_value=0, max_value=400, value=265)
ASPs[4]=st.sidebar.slider('Area 5',min_value=0, max_value=400, value=265)
ASPs[5]=st.sidebar.slider('Area 6',min_value=0, max_value=400, value=265)
ASPs[6]=st.sidebar.slider('Area 7',min_value=0, max_value=400, value=265)
ASPs[7]=st.sidebar.slider('Area 8',min_value=0, max_value=400, value=265)
ASPs[8]=st.sidebar.slider('Area 9',min_value=0, max_value=400, value=265)
ASPs[9]=st.sidebar.slider('Area 10',min_value=0, max_value=400, value=265)
ASPs[10]=st.sidebar.slider('Area 11',min_value=0, max_value=400, value=265)
ASPs[11]=st.sidebar.slider('Area 12',min_value=0, max_value=400, value=265)
ASPs[12]=st.sidebar.slider('Area 13',min_value=0, max_value=400, value=290)
ASPs[13]=st.sidebar.slider('Area 14',min_value=0, max_value=400, value=290)

if st.sidebar.button('Run Model'):
    df_in=updatedf(ASPs,dfbase)
    results_df=RunModel(df_in)
    Predicted_coulombs= [results_df['Front_Amps'].mul(2).sum(),
                         results_df['Roof_Amps'].mul(2).sum(),
                         results_df['RHWing_Amps'].mul(2).sum(),
                         results_df['LHWing_Amps'].mul(2).sum(),
                         results_df['Rear_Amps'].mul(2).sum(),
                         results_df['Body_Amps'].mul(2).sum()]
    results_available=True


#----visualise the results-------#
if (results_available):
    #option = st.selectbox('Display',results_df.columns)

    n_groups=6
    bar_width = 0.25
    opacity=0.8
    index = np.arange(n_groups)
    plt.bar(index,Predicted_coulombs,bar_width,alpha=opacity,label='Predicted')
    plt.bar(index+bar_width,BaseLine_coulombs,bar_width,alpha=opacity,label='Base Line')
    plt.legend()
    plt.xlabel('Panel Location')
    plt.ylabel('Coulombs (A.sec)')
    plt.xticks(index, ('Front', 'Roof', 'RHWing', 'LHWing','Rear','Body'))
    plt.tight_layout()
    plt.grid(linestyle='-', linewidth=0.5)
    plt.title('Coulombs',fontsize=10)
    st.pyplot()
    plt.clf()

    ' '
    
    plt.plot(results_df.index,results_df.loc[:,'Body_Amps'],label='Predicted')
    if show_baseline:
        plt.plot(results_df.index,df_BaseLine.loc[:,'Body_Amps'],label='Base line')
    plt.title('Predict Amps on Body')
    plt.legend()
    st.pyplot()
    plt.clf()

    plt.plot(results_df.index,results_df.loc[:,'Front_Amps'],label='Predicted')
    if show_baseline:
        plt.plot(results_df.index,df_BaseLine.loc[:,'Front_Amps'],label='Base line')
    plt.title('Predict Front Chassis Amps')
    plt.legend()
    st.pyplot()
    plt.clf()

    ' '

    plt.plot(results_df.index,results_df.loc[:,'Roof_Amps'],label='Predicted')
    if show_baseline:
        plt.plot(results_df.index,df_BaseLine.loc[:,'Roof_Amps'],label='Base line')
    plt.title('Predict Roof Amps')
    plt.legend()
    st.pyplot()
    plt.clf()

    ' '
    
    plt.plot(results_df.index,results_df.loc[:,'RHWing_Amps'],label='Predicted')
    if show_baseline:
        plt.plot(results_df.index,df_BaseLine.loc[:,'RHWing_Amps'],label='Base line')
    plt.title('Predict Amps RH Wing Mount')
    plt.legend()
    st.pyplot()
    plt.clf()

    ' '
    
    plt.plot(results_df.index,results_df.loc[:,'LHWing_Amps'],label='Predicted')
    if show_baseline:
        plt.plot(results_df.index,df_BaseLine.loc[:,'LHWing_Amps'],label='Base line')
    plt.title('Predict Amps LH Wing Mount')
    plt.legend()
    st.pyplot()
    plt.clf()

    ' '
    
    
    plt.plot(results_df.index,results_df.loc[:,'Rear_Amps'],label='Predicted')
    if show_baseline:
        plt.plot(results_df.index,df_BaseLine.loc[:,'Rear_Amps'],label='Base line')
    plt.title('Predict Amps Rear Chassis')
    plt.legend()
    st.pyplot()
    plt.clf()

    if show_resultsdf:
        st.write(results_df)
    
    if show_spdf:
        st.write(df_in)


