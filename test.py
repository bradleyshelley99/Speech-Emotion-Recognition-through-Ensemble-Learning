def test(data, labels):
    import numpy as np
    from joblib import load
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from tensorflow import keras
    import librosa
    from librosa.feature import melspectrogram
    from librosa.util import normalize as norm
    
    #Chromagram feature extraction
    def chroma(Input,sample_rate):
        stft=np.abs(librosa.stft(Input))
        value =np.array([])
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        value = np.hstack((value,chroma))
        return value
        
    #Data1 processing with MFCC feature extraction
    def data_1():
        datamfcc = []
        for i in range(len(data)):
            k=librosa.feature.mfcc(y=data[i,:], sr=44000)
            temp = []
            for j in range(20):
                temp += [np.mean(k[j,:])]
            datamfcc += [temp]
        datachroma = []
        for i in range(len(data)):
            k=chroma(data[i,:],44000)
            datachroma += [k]
        #Normalize and stack data
        datamfcc = np.array(datamfcc)
        datachroma = np.array(datachroma)
        datamfcc = norm(datamfcc)
        Xdata = np.hstack((datamfcc, datachroma))
        return Xdata
        
    #Data2 processing (spectrograms)
    def data_2():
        spec = []
        for i in range(data.shape[0]):
            spectrogram = melspectrogram(data[i,:],sr=44000,n_fft=1024,win_length = 512,window='hamming',hop_length = 256,n_mels=128, fmax = 22000)
            spectrogram = librosa.power_to_db(spectrogram)
            spectrogram = norm(spectrogram)
            spec += [spectrogram]
        spec = np.array(spec)
        spec = spec.reshape(len(spec),128,391,1)
        return spec
    
    #Shuffling inputs
    shuffler = np.random.permutation(len(data))
    data = data[shuffler]
    labels = labels[shuffler]
    
    #Data processing and feature extraction
    data1 = data_1()
    data2 = data_2()
    
    #Load saved models and PCA transformations
    KNNmodel = load('KNN_model.joblib')
    KNNpca = load('KNN_PCA.joblib')
    MLPmodel = load('MLP_model.joblib')
    MLPpca = load('MLP_PCA.joblib')
    CNNmodel = keras.models.load_model('CNN_model')
    
    #Make predictions with test data
    scores = []
    KNNpred = KNNmodel.predict(KNNpca.transform(data1))
    KNNprob = KNNmodel.predict_proba(KNNpca.transform(data1))
    scores += [accuracy_score(KNNpred,labels)]
    MLPpred = MLPmodel.predict(MLPpca.transform(data1))
    MLPprob = MLPmodel.predict_proba(MLPpca.transform(data1))
    scores += [accuracy_score(MLPpred,labels)]
    CNNprob = CNNmodel.predict(data2)
    CNNprob = CNNprob[:,1:]
    CNNpred = np.argmax(CNNprob, axis=1)
    CNNpred = CNNpred + 1
    scores += [accuracy_score(CNNpred,labels)]
    
    #Ensemble learning/voting system
    final_pred = []
    for i in range(len(KNNpred)):
        if CNNpred[i]==KNNpred[i] and KNNpred[i]==MLPpred[i]:
            final_pred += [CNNpred[i]]
        elif CNNpred[i]==KNNpred[i] or KNNpred[i]==MLPpred[i] or CNNpred[i]==MLPpred[i]:
            if CNNpred[i]==KNNpred[i] or CNNpred[i]==MLPpred[i]:
                final_pred += [CNNpred[i]]
            elif KNNpred[i]==CNNpred[i] or KNNpred[i]==MLPpred[i]:
                final_pred += [KNNpred[i]]
            elif MLPpred[i]==CNNpred[i] or MLPpred[i]==KNNpred[i]:
                final_pred += [MLPpred[i]]
            else:
                print('bug')
        else:
            prob = CNNprob[i] + MLPprob[i] + KNNprob[i]
            pred = np.argmax(prob)
            pred = pred+1
            final_pred += [pred]
    
    #Outputs
    final_pred = np.array(final_pred)
    score = accuracy_score(final_pred,labels)
    cm = confusion_matrix(final_pred,labels)
    return score,cm,final_pred,scores
