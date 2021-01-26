def train(data_training,labels_training):
    import numpy as np
    from numpy import random
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    import librosa
    from librosa.feature import melspectrogram
    from librosa.util import normalize as norm
    from IPython.display import clear_output
    from tensorflow.keras import models
    from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D 
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.layers import Dropout
    from joblib import dump
    import warnings
    warnings.filterwarnings('ignore')
    
    #Update progress bar for feature extration and data processing
    #Reference: https://stackoverflow.com/questions/3160699/python-progress-bar/15860757#15860757
    def update_progress(progress):
        bar_length = 20
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
        if progress < 0:
            progress = 0
        if progress >= 1:
            progress = 1
        block = int(round(bar_length * progress))
        clear_output(wait = True)
        text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
        print(text)

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
        for i in range(len(data_training)):
            k=librosa.feature.mfcc(y=data_training[i,:], sr=44000)
            temp = []
            for j in range(20):
                temp += [np.mean(k[j,:])]
            datamfcc += [temp]
            update_progress(i / len(data_training))
        datachroma = []
        for i in range(len(data_training)):
            k=chroma(data_training[i,:],44000)
            datachroma += [k]
            update_progress(i / len(data_training))
        #Normalize and stack data
        datamfcc = np.array(datamfcc)
        datachroma = np.array(datachroma)
        datamfcc = norm(datamfcc)
    
        Xdata = np.hstack((datamfcc, datachroma))
        return Xdata
        
    #Data2 processing (spectrograms)
    def data_2():
        data = []
        for i in range(data_training.shape[0]):
            spectrogram = melspectrogram(data_training[i,:],sr=44000,n_fft=1024,win_length = 512,window='hamming',hop_length = 256,n_mels=128, fmax = 22000) 
            spectrogram = librosa.power_to_db(spectrogram)
            spectrogram = norm(spectrogram)
            data += [spectrogram]
            update_progress(i / len(data_training))
        data = np.array(data)
        return data
    
    #Select parameter and train k-NN Classifier
    def KNN_Classifier(Xtrain1, Ytrain1, Xtest1, Ytest1):
        print('training KNN...')
        
        #4-Fold validation to select k neighbors
        kf = KFold(n_splits=4)
        kvals = np.arange(1,50,1)
        maxacckNN = 0
        maxk = 0
        for k in kvals:
            acc  = 0
            f=1
            for train_index, validation_index in kf.split(Xtrain1):
                xt, xv = Xtrain1[train_index,:], Xtrain1[validation_index,:]
                yt, yv = Ytrain1[train_index], Ytrain1[validation_index]
    
                neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=(k),weights='distance')
                neigh.fit(xt, yt)
                yKNN = neigh.predict(xv)
    
                acc += sklearn.metrics.accuracy_score(yv,yKNN)
                f+=1  
            if (acc>maxacckNN):
                maxacckNN = acc
                maxk = k
                
        #Select PCA components
        maxaccPCA = 0
        maxC = 0
        for i in range(31):
            pca = PCA(n_components=i+1)
            pca.fit(Xtrain1)
            neighpca = KNeighborsClassifier(n_neighbors=maxk)
            neighpca.fit(pca.transform(Xtrain1), Ytrain1)
            acc_pca = neighpca.score(pca.transform(Xtest1), Ytest1)
            if maxaccPCA<acc_pca:
                maxaccPCA = acc_pca
                maxC = i+1
        
        #PCA transformation and train k-NN model
        pca = PCA(n_components=maxC)
        pca.fit(data1)
        model = KNeighborsClassifier(n_neighbors=maxk,weights='distance')
        model.fit(pca.transform(data1), labels_training)
        
        #Save model and PCA transformation
        dump(model, 'KNN_model.joblib')
        dump(pca, 'KNN_PCA.joblib')
    
    #MLP Classifier
    def MLP_Classifier(Xtrain1, Ytrain1, Xtest1, Ytest1):
        print('training MLP...')
        
        #Determine the best PCA components
        score = 0
        scoreidx = 0
        for i in range(31):
            pca = PCA(n_components=i+1)
            pca.fit(Xtrain1)
            model = MLPClassifier(hidden_layer_sizes=(1000,1000),verbose=False, activation = 'relu', solver='adam',learning_rate_init=0.01, max_iter=200).fit(pca.transform(Xtrain1), Ytrain1)
            temp = model.score(pca.transform(Xtest1), Ytest1)
            if temp>score:
                score = temp
                scoreidx = i+1
                
        #PCA transformation and train MLP model
        pca = PCA(n_components=scoreidx)
        pca.fit(data1)
        model = MLPClassifier(hidden_layer_sizes=(1000,1000),verbose=False, activation = 'relu', solver='adam',learning_rate_init=0.01, max_iter=200).fit(pca.transform(data1), labels_training)
        
        #Save model and PCA transformation
        dump(model, 'MLP_model.joblib')
        dump(pca, 'MLP_PCA.joblib')
        
    #CNN Classifier
    def CNN_Classifier():
        print('training CNN...')
        Xtrain2 = data2.reshape(len(data2),128,391,1)
        Ytrain2 = to_categorical(labels_training)
        model = models.Sequential()
        model.add(Conv2D(32, (3,3), activation='relu',padding='same', input_shape=(128,391,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(9, activation='softmax'))
        model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(Xtrain2, Ytrain2, epochs=75)
        #Save model
        model.save('CNN_model')
        
    #Shuffle input
    shuffler = np.random.permutation(len(data_training))
    data_training = data_training[shuffler]
    labels_training = labels_training[shuffler]
    
    #Data processing and feature extraction
    print('Extracting feature 1...')
    data1 = data_1()
    print('Extracting feature 2...')
    data2 = data_2()
    
    #Train test split for parameter selection
    rs = random.randint(0,43)
    Xtrain1, Xtest1, Ytrain1, Ytest1 =  train_test_split(data1, labels_training, test_size=0.2, random_state=rs)    
    Xtrain2, Xtest2, Ytrain2, Ytest2 =  train_test_split(data2, labels_training, test_size=0.2, random_state=rs)
    
    #Train classifiers
    KNN_Classifier(Xtrain1, Ytrain1, Xtest1, Ytest1)
    MLP_Classifier(Xtrain1, Ytrain1, Xtest1, Ytest1)
    CNN_Classifier()
