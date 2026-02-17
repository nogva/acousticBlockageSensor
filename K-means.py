import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from glob import glob
from scipy import signal
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans
import seaborn as sns


def splitFolders(path):
    return [ sub_Dir_Name.split('/')[-1] for sub_Dir_Name in glob(path+'/*',recursive=False) ]

def createDF(columns,DataTypeList):
    """
    Creates a DataFrame and generates mapping dictionaries for categorizing data types.
    This function initializes a pandas DataFrame with specified columns and creates
    four dictionary mappings that categorize items from DataTypeList by different
    criteria:
    - IdDict: Maps each unique data type name to a sequential ID
    - IdDict3: Maps the prefix of each data type (before first underscore) to a sequential ID
    - IdDict2: Maps the suffix of each data type (after first underscore) to a sequential ID
    If IS_MONO is False, returns two DataFrames; otherwise returns a single DataFrame.
    Args:
        columns (list): Column names for the DataFrame(s)
        DataTypeList (list): List of data type names to be categorized
    Returns:
        If IS_MONO is True:
            tuple: (df, IdDict, IdDict2, IdDict3)
                - df (pd.DataFrame): Empty DataFrame with specified columns
                - IdDict (dict): Maps full data type names to IDs
                - IdDict2 (dict): Maps data type suffixes to IDs
                - IdDict3 (dict): Maps data type prefixes to IDs
        If IS_MONO is False:
            tuple: ([df, df2], IdDict, IdDict2, IdDict3)
                - [df, df2] (list): List of two empty DataFrames with specified columns
                - IdDict (dict): Maps full data type names to IDs
                - IdDict2 (dict): Maps data type suffixes to IDs
                - IdDict3 (dict): Maps data type prefixes to IDs
    """

    df=pd.DataFrame(columns=columns)
    IdDict={}
    IdDict2={}
    IdDict3={}

    ID=0
    ID2=0
    ID3=0
  
    for name in DataTypeList:
        if name not in IdDict:
            IdDict[name]=ID
            ID+=1
        if name.split('w')[0] not in IdDict3:
            IdDict3[name.split('w')[0]]=ID3
            ID3+=1
        b=name.split('w')
        if len(b)==1:
            b='not'
        else:
            b=b[1]
        if  b not in IdDict2:
            IdDict2[b]=ID2
            ID2+=1
    
    if not IS_MONO:
        df2=pd.DataFrame(columns=columns)
        return [df,df2],IdDict,IdDict2,IdDict3
    return df, IdDict,IdDict2,IdDict3

def bandpass_filter(y, sr):
    low = 50
    high = 15000
    sos = signal.butter(
        4,                 # order (4 is good)
        [low, high],       # frequency band
        btype='bandpass',
        fs=sr,
        output='sos'
    )
    return signal.sosfilt(sos, y)

def readData(path,df_list,IdDict,IdDict2,IdDict3): 
    if IS_MONO:
        df=df_list[0]
        x_fs_list=[]
        i=0 
        for audioFile in glob(path+'/*'+'/*'):
            name=audioFile.split('/')[-2]
            x,fs=librosa.load(audioFile,sr=None)
            x_fs_list.append((x[:5*fs],fs)) 
            df.at[i,'type']=name
            df.at[i,'ID']=IdDict[name]

            df.at[i,'ID3']=IdDict3[name.split('_')[0]]
            df.at[i,'type3']=name.split('_')[0]

            if len(name.split('_'))==2:
                df.at[i,'ID2']=IdDict2[name.split('_')[1]]
                df.at[i,'type2']=name.split('_')[1]
            else:
                df.at[i,'ID2']=IdDict2['not']
                df.at[i,'type2']='not'
            i+=1
        return x_fs_list
    elif not IS_MONO:
        # lag to lister
        x_fs_list_B=[]
        x_fs_list_R=[]
        i=0
        for audioFile in glob(path+'/*'+'/*'):
            name=audioFile.split('/')[-2]
            x,fs=librosa.load(audioFile,mono=IS_MONO,sr=None)
            x_fs_list_B.append((bandpass_filter(x[0][2:28*fs],fs),fs)) 
            x_fs_list_R.append((bandpass_filter(x[1][2:28*fs],fs),fs)) 
            df_list[0].at[i,'type']=name
            df_list[0].at[i,'Pos']='Black'
            df_list[0].at[i,'ID']=IdDict[name]
            
            df_list[1].at[i,'type']=name
            df_list[1].at[i,'Pos']='Red'
            df_list[1].at[i,'ID']=IdDict[name]

            df_list[0].at[i,'ID3']=IdDict3[name.split('w')[0]]
            df_list[0].at[i,'type3']=name.split('w')[0]
            df_list[1].at[i,'ID3']=IdDict3[name.split('w')[0]]
            df_list[1].at[i,'type3']=name.split('w')[0]


            if len(name.split('w'))==2:
                df_list[0].at[i,'ID2']=IdDict2[name.split('w')[1]]
                df_list[0].at[i,'type2']=name.split('w')[1]
                df_list[1].at[i,'ID2']=IdDict2[name.split('w')[1]]
                df_list[1].at[i,'type2']=name.split('w')[1]

            # else:
            #     df_list[0].at[i,'ID2']=IdDict2['not']
            #     df_list[0].at[i,'type2']='not'
            #     df_list[1].at[i,'ID2']=IdDict2['not']
            #     df_list[1].at[i,'type2']='not'

            i+=1
        return [x_fs_list_B,x_fs_list_R]
    else:
        print('Something worng')
        return



def zcr(x,fs):
    return np.mean(librosa.zero_crossings(x))

def energy(x,fs):
    return scipy.linalg.norm(x)

def extract_mfcc(signal,fs):
    # n_mfcc=50 #<- this may be changed
    mfcc=librosa.feature.mfcc(y=signal, sr=fs)
    mfccs = np.mean(mfcc.T,axis=0)
    return mfccs

def extract_spectral_centroid(signal,fs):
    return np.mean(librosa.feature.spectral_centroid(y=signal, sr=fs))

def spec_bw(signal,fs):
    return np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=fs))

def extract_stft_mag(signal,fs):
    return np.mean(np.abs(librosa.amplitude_to_db(librosa.stft(signal,n_fft=2048,hop_length=512),ref=np.max)).T,axis=0)

def extract_stft_phase(signal,fs):
    return np.mean(librosa.feature.poly_features(S=np.angle(librosa.amplitude_to_db(librosa.stft(signal,n_fft=2048,hop_length=512),ref=np.max)),order=2).T,axis=0)

def chroma_stf(signal,fs):
    return np.mean(librosa.feature.chroma_stft(y=signal, sr=fs))

def rmse(signal,fs):
    return np.mean(librosa.feature.rms(y=signal))

def SpecRolloff(signal,fs):
    return np.mean(librosa.feature.spectral_rolloff(y=signal, sr=fs))

def extract_features(x, fs,**kvargs):
    names=np.array(list(kvargs.keys()))
    return names, [kvargs[name](x,fs) for name in names]
    
def subFunction_plotRealData(features_scaled,df,names,axis=(None,None),ax=None):
    if None not in axis:
        for i in range(len(names)):
            name=names[i]
            df_relevant=df.loc[df['type']==name]
            x_vals=df_relevant[axis[0]].to_numpy()
            y_vals=df_relevant[axis[1]].to_numpy()
            id_arr=df['ID'].to_numpy().astype(int)
            ax.scatter(x_vals, y_vals,label='Class'+name)
        ax.set_xlabel(axis[0])
        ax.set_ylabel(axis[1])
    else:
        for i in range(len(names)):
            name=names[i]
            id_arr=df['ID'].to_numpy().astype(int)
            ax.scatter(features_scaled[id_arr==i,0], features_scaled[id_arr==i,1],label='Class'+name)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    ax.set_title('RealData_'+df.at[1,'Pos'])
    ax.legend(loc='upper right')


def plotRealData(features_scaled,df,names,axis=(None,None)):
    if IS_MONO:
        fig,axes=plt.figure()
        fig.add_subplot(212)(subFunction_plotRealData(features_scaled,df,names,axis,axes[0]))
    elif not IS_MONO:
        fig,axes=plt.subplots(ncols=2,nrows=1)
        subFunction_plotRealData(features_scaled[0],df[0],names,axis,axes[0])
        subFunction_plotRealData(features_scaled[1],df[1],names,axis,axes[1])
        
    return fig



def fillFeaturList(x_fs_list,df,**kvargs):
    features_list=[]
    for i in range(len(x_fs_list)):
        x_arr=x_fs_list[i][0]
        fs=x_fs_list[i][1]
        names,features = extract_features(x_arr/np.max(np.abs(x_arr)),fs,**kvargs)

        for nameIndex,feature in enumerate(features):
            if(type(feature) is not np.ndarray):
                df.at[i,names[nameIndex]]=feature
            else:
                for j in range(len(feature)):
                    df.at[i,names[nameIndex]+str(j)]=feature[j]

        flat_features = np.hstack([np.atleast_1d(f) for f in features])
        features_list.append(flat_features)
    features_arr=np.squeeze(np.array(features_list))
    return features_arr

def create_feature_list(x_fs_list,df_list,**kvargs):
    if IS_MONO:
        return fillFeaturList(x_fs_list,df_list,**kvargs)
    elif not IS_MONO:
        return [fillFeaturList(x_fs_list[0],df_list[0],**kvargs),fillFeaturList(x_fs_list[1],df_list[1],**kvargs)]





def testAccuarcy(dataFrame,whichID='ID',nameType='type'):

    
    corrID=dataFrame[whichID].to_numpy().astype(np.int64)
    lab=dataFrame['label'].to_numpy().astype(np.int64)
    name=dataFrame[nameType].to_numpy()
    ############################
    assert lab.size == corrID.size
    D = max(lab.max(), corrID.max())+1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(lab.size):
        w[lab[i], corrID[i]] += 1
    ind = linear_assignment(w.max() - w)
    newSum=np.zeros_like(ind[1])
    
    for i in range(len(newSum)):
        newSum[i]=w[i,ind[1][i]]
    print(newSum)

    ##############################

    y=np.zeros(shape=(len(set(name)),len(set(name))))
    # correct=0
    # miss=0
    names=[]

    


    for i in range(len(corrID)):
        if name[i] not in names:
            names.append(name[i])
        y[int(corrID[i]),ind[1][int(lab[i])]]+=1
        
        
    #     if int(corrID[i])==ind[1][int(lab[i])]:
    #         correct+=1
    #     else:
    #         miss+=1
    
    

    # df_cm = pd.DataFrame(y, index = [i for i in names],
    #               columns = [i for i in names])
    plt.figure(figsize = (10,7))
    sns.heatmap(w, annot=True)



    # print('correct: {}, miss: {}\n'.format(correct,miss))
    # print('Accuracy: {:%}\n'.format(sum(newSum)/lab.size))

    # for i in range (len(names)):
    #     print('{:<15}:\t {} correct and {} miss'.format(names[i],y[i,i],y[i].sum()-y[i,i]))

def elbowPLot(features_scaled,it):
    clusters = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i).fit(features_scaled)
        clusters.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
    ax.set_title('--- Searching for Elbow ---')
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')
    return fig

def subFunction_plot2DKmeans(features_scaled,labels,nCluster,ax,tit=None):
    for i in range(nCluster):
        ax.scatter(features_scaled[labels==i,0], features_scaled[labels==i,1], c=color[i],label='Class'+str(i))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if tit != None:
        ax.set_title('2D-Kmeans_'+tit)
    ax.legend()
    
def plot2DKmeans(features_scaled,labels,nCluster,dfs=None):
    if IS_MONO:
        fig,axes=plt.figure()
        if dfs != None:
            tit=dfs.at[1,'Pos']
        subFunction_plot2DKmeans(features_scaled,labels,nCluster,axes[0],tit)
    elif not IS_MONO:
        fig,axes=plt.subplots(ncols=2,nrows=1)
        if dfs != None:
            tit1=dfs[0].at[1,'Pos']
            tit2=dfs[1].at[1,'Pos']

        subFunction_plot2DKmeans(features_scaled[0],labels[0],nCluster,axes[0],tit1)
        subFunction_plot2DKmeans(features_scaled[1],labels[1],nCluster,axes[1],tit2)
    return fig

def plot3DKmeans(features_scaled,labels,nCluster):
    fig =plt.figure(figsize=(14,14))
    ax = fig.add_subplot(projection='3d')
    for i in range(nCluster):
        ax.scatter(features_scaled[labels==i,0], features_scaled[labels==i,1], features_scaled[labels==i,2], c=color[i],label='Class'+str(i))
    

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    plt.legend()
    
    return fig,ax


def plotKmeans(features_scaled,nCluster,dim=2,df=None):
    if dim not in (2,3):
        print('Dimention need to be 2 or 3. dim=2 if not specfied.')
        return None
    model = sklearn.cluster.KMeans(n_clusters=nCluster)
    if IS_MONO:
        
        labels = model.fit_predict(features_scaled)
        if df is not None:
            for i, lab in enumerate(labels):
                df.at[i,'label']=lab
        
        if dim==2:
            plot2DKmeans(features_scaled,labels,nCluster)
            plt.title('2D-Kmeans')
        elif dim==3:
            fig,ax=plot3DKmeans(features_scaled,labels,nCluster)
            ax.set_title('3D-Kmeans')


    elif not IS_MONO:
        labels_1 = model.fit_predict(features_scaled[0])
        labels_2 = model.fit_predict(features_scaled[1])
        if df is not None:
            for i, lab1 in enumerate(labels_1):
                df[0].at[i,'label']=lab1
            for i, lab2 in enumerate(labels_2):
                df[1].at[i,'label']=lab2
        if dim==2:

            fig=plot2DKmeans(features_scaled,[labels_1,labels_2],nCluster,dfs=df)
            
        elif dim==3:
            fig,ax=plot3DKmeans(features_scaled,labels,nCluster)
            ax.set_title('3D-Kmeans')
    return fig


color=['b','g','r','c','k','m']

Featurs_dict={}


master_dir = 'data\3m straight pipe\first_test'

IS_MONO=False

sampFreq=48000 #Hz

bitRes=24

name_list=splitFolders(master_dir)

# print(name_list)

df_list, IdDict,IdDict2,IdDict3=createDF([],name_list)

# print(df_list,IdDict,IdDict2,IdDict3)

x_fl_2Dlist=readData(master_dir,df_list,IdDict,IdDict2,IdDict3)



Featurs_dict['zcr']=zcr
Featurs_dict['energy']=energy
Featurs_dict['mfcc']=extract_mfcc
# Featurs_dict['Rolloff']=SpecRolloff
# Featurs_dict['specCen']=extract_spectral_centroid
# Featurs_dict['specBW']=spec_bw
# Featurs_dict['chromaStf']=chroma_stf
# Featurs_dict['rmse']=rmse
# Featurs_dict['stft_phase']=extract_stft_phase

feature_arr_2d=create_feature_list(x_fl_2Dlist,df_list,**Featurs_dict)


fig=plotRealData(feature_arr_2d ,df_list,name_list)
plt.show()


# # ####
fig2=plotKmeans(feature_arr_2d ,5,2,df_list)
plt.show()


acc=testAccuarcy(df_list[0])
plt.show()
 



if __name__ == '__main__':
    pass