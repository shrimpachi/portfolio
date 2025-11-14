#%%
import numpy as np
import requests
import io,time,os,sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_settings():
    use_cluster = True
    epochs = 50
    learning_rate = 0.00001
    loss_function = 'binary_crossentropy'
    batch_size = 333 #per label
    amount = 666  #per label
    folder_name = "One_Batch_V5_diff_model_50"

    print("Learning Rate:",learning_rate)
    print("Loss Function:",loss_function)
    print("Epochs:",epochs)
    print("Batch size:",batch_size)
    print("Amount of rows per label:",amount)

    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # You are using GPU
        print("GPU is available")
    else:
        # You are using CPU
        print("GPU is not available")

    return use_cluster,epochs,learning_rate,loss_function,batch_size,amount,folder_name

#%%
    
def update_progress_file(update:str,progress_filename):
    try:
        progress_file = open(progress_filename, "a")
    except:
        print("Progress update failed. Couldn't open the file to append.")
    try:
        progress_file.write(update + "\n")
    except:
        print("Progress update failed. Couldn't append to the file.")
    progress_file.close()
    

def get_file(num:str, label:int, detector:str):
    if len(num)==1:
        num= "00"+str(num)
    elif len(num)==2:
        num="0"+str(num)
    url = "https://ldas-jobs.ligo.caltech.edu/~melissa.lopez/ML_course_mock/EarlyWarning/mock_data/0/00000"+str(num)+"_"+str(label)+"_"+detector+".npy"
    response = requests.get(url)
    response.raise_for_status()
    rawdata=np.load(io.BytesIO(response.content))
    
    return rawdata

def get_file_cluster(num:str, label:int, detector:str):
    path = "/dcache/gravwav/lopezm/ML_projects/Projects_GW/EarlyWarning/Data/FreqCL/0/"
    if len(num) == 1:
        num = "000" + str(num)
    elif len(num) == 2:
        num = "00" + str(num)
    elif len(num) == 3:
        num = "0"+ str(num)
    elif len(num) == 4:
        num = str(num)
    filename = "0000" + str(num) + "_" + str(label) + "_" + str(detector) + ".npy"
    rawdata = np.load(path+filename)
    if num == 0: print(np.shape(rawdata))
    
    return rawdata


#amount = 10

def load_batch_data(batch_size,start_punt,use_cluster):
    num = np.linspace(start_punt, start_punt+batch_size-1,batch_size, dtype='int32')
    labels = [0,1]
    detectors = ["H1","L1","V1"]
    data = []
    for n in num:
        rawdata = []
        for label in labels:
            rawdata2=[]
            for det in detectors:
                if use_cluster:
                    rawdata2.append(np.asarray(get_file_cluster(str(n), label, det)))
                else:
                    rawdata2.append(np.asarray(get_file(str(n), label, det)))
                #print("Appended number " + str(n) + " label " + str(label) +  " detector " +det+ " to the rawdata.")
            rawdata.append(rawdata2)
        data.append(rawdata)
    #print(np.shape(data))
    #Transpose data to shape (amount, len(labels),timesteps,len(detectors))
    #I.e. draai de laatste twee om, zdd de input aansluit op de CNN architectuur
    data = np.transpose(data, (0, 1, 3, 2))
    #print(np.shape(data))

    #%Preprocessing data to feature and label arrays
    data_0 = [item[0] for item in data]
    data_1 = [item[1] for item in data]
    features = np.concatenate([data_0,data_1],axis=0).tolist()
    labels = [[1.0,0.0]]*batch_size + [[0.0,1.0]]*batch_size

    print("Shape of Features:")
    print(np.shape(features))
    # print(np.shape(features)[0], "is the amount of input rows")
    # print(np.shape(features)[1], "is the amount of time steps (155648)")
    # print(np.shape(features)[2], "is the amount of channels i.e. detectors (should be 3)")
    # print("Shape of Labels:")
    #print(np.shape(labels))
    X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, random_state=42,shuffle=True)
    #print("Train/Test sets gemaakt")
    return X_train,X_test,y_train,y_test


def make_folder(folder_name,ROOT_DIR):
    folder_tot = os.path.join(ROOT_DIR,folder_name)
    try:
        os.mkdir(folder_tot)
    except:
        pass
    return folder_tot

#%% Define model
    

def model_init(learning_rate,loss_function):  
    steps = 155648
    channels = 3
    # steps = np.shape(features)[1]
    # channels = np.shape(features)[2]
    model = models.Sequential()

    #Fast
    model.add(layers.Conv1D(32,128, activation='relu', input_shape=(steps,channels)))
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Conv1D(64, 32, activation='relu'))
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Conv1D(32, 16, activation='relu'))
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Conv1D(16, 32, activation='relu'))
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Conv1D(8, 128, activation='relu'))
    model.add(layers.MaxPooling1D(4))   

    #Fully connected layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))

    #Output layer
    model.add(layers.Dense(2, activation='softmax'))
    print(model.summary())
    opt = Adam(learning_rate=learning_rate)#,weight_decay=0.00001)
    print("Optimizer aangemaakt")
    model.compile(optimizer=opt,
                loss=loss_function,
                metrics=['accuracy'])
    print("Model gecompileerd")
    return model
#%% Train and validate model

def train_model_batch(batch_size,model,X_train,y_train,X_test,y_test,final_epoch=False):
    #print("Probeer te beginnen met trainen")

    #history = model.fit(X_train, y_train, epochs=25, batch_size=batch_size,
                        #validation_data=(X_test,y_test))
                        
    history = model.fit(X_train, y_train, epochs=1,
                        validation_data=(X_test,y_test),batch_size=batch_size)
    #print("Model getraind")
    if final_epoch:
        train_pred = model.predict(X_train)#.flatten()
        test_pred = model.predict(X_test)#.flatten()
    else:
        train_pred,test_pred=0,0#placeholder

    #print("Train and Test Predictions done")
    return model,history,train_pred,test_pred
#%%
def plot_results(history,y_train,y_test,train_pred,test_pred,ROOT_DIR):
    # Plot training history
    plt.figure(figsize=(12, 4))
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.tight_layout()
    try:
        plt.savefig(ROOT_DIR+"/SkillGraphs.png")
    except FileNotFoundError:
        plt.savefig("../Output_J/SkillGraphs.png")
    plt.show()

    print("Training and Validation history plotted")

    #Convert labels to binary (single-column) for confusion matrix plotting
    y_train_b_tus = [y_train[i][0] for i in range(len(y_train))]
    y_test_b_tus = [y_test[i][0] for i in range(len(y_test))]
    y_train_b = (np.array(y_train_b_tus) > 0.5).astype(int)
    y_test_b = (np.array(y_test_b_tus)> 0.5).astype(int)
    train_pred_b = (train_pred[:,0]> 0.5).astype(int)
    test_pred_b = (test_pred[:,0] > 0.5).astype(int)

    def plot_confusion_matrix(cm, classes, title,set):
        plt.figure(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        try:
            plt.savefig(ROOT_DIR+"/ConfMat_"+set+".png")
        except FileNotFoundError:
            plt.savefig("../Output_J/ConfMat_"+set+".png")
        plt.show()

    train_cm = confusion_matrix(y_train_b, train_pred_b)
    test_cm = confusion_matrix(y_test_b, test_pred_b)
    plot_confusion_matrix(train_cm, classes=range(2), title='Training Confusion Matrix',set="train")
    plot_confusion_matrix(test_cm, classes=range(2), title='Validation Confusion Matrix',set="test")

    print("Confusion Matrices plotted")

#Deze functie wordt automatisch aangeroepen bij het draaien van het script
if __name__ == "__main__":
    starting_time = time.time()
    try:
        ROOT_DIR = str(sys.argv[1])
    except IndexError:
        ROOT_DIR = r'C:\Users\JFK-9\Documents\_Studie\CA Machine Learning\Lokale_Output'

    #Druk de configuratie af i.e. learning rate en zo
    use_cluster,epochs,learning_rate,loss_function,batch_size,amount,folder_name=print_settings()
    ROOT_DIR = make_folder(folder_name,ROOT_DIR)
    amount_of_batches = amount//batch_size
    #Initieer het CNN op basis van de parameters
    model = model_init(learning_rate,loss_function)

    #Aantal iteraties die nodig zijn obv aantal rijen en batch grootte
    #Itereer over de batches; start_punt houdt de index van de .npy bestanden in de folder bij
    #voor elke batch:
    # - lees de data in met load_batch_data; zelfde structuur als eerder met get_file_cluster, data.append enzovoort
    # - train het model met deze data, train_model_batch. de 'model' variabele wordt gereturned, hierin zitten ook de geupdatete weights
    all_histories = []
    for i in range(epochs-1): #epochs-1, laatste gaat apart zdd resultaten opgeslagen kunnen worden
        print("Start Epoch",i+1)
        start_punt=0
        epoch_hists=[]

        #
        for k in range(amount_of_batches):
            print("Startpunt:",start_punt)
            X_train,X_test,y_train,y_test=load_batch_data(batch_size,start_punt,use_cluster)
            model,history,train_pred,test_pred=train_model_batch(batch_size,model,X_train,y_train,X_test,y_test)
            start_punt+=batch_size
            epoch_hists.append(history)
            print("Batch",k+1,'verwerkt')

        #save the results of this epoch   
        combined_epoch_history = {}
        for key in epoch_hists[0].history.keys():
            combined_epoch_history[key] = np.concatenate([h.history[key] for h in epoch_hists], axis=0)
        all_histories.append(combined_epoch_history)
        print("Epoch",i+1,"klaar, tussentijd:",time.time()-starting_time)
        update_progress_file("Epoch "+str(i+1)+"; time: "+str(time.time()-starting_time),ROOT_DIR+"/updateje.txt")

    #Laatste Epoch, sla resultaten op
    print("Start final Epoch")
    y_trains,y_tests,train_preds,test_preds=[],[],[],[]
    y_train_tot=[]
    y_test_tot=[]
    train_pred_tot = []
    test_pred_tot=[]
    epoch_hists=[]
    start_punt=0
    for _ in range(amount_of_batches):
        print("Startpunt:",start_punt)
        X_train,X_test,y_train,y_test=load_batch_data(batch_size,start_punt,use_cluster)
        model,history,train_pred,test_pred=train_model_batch(batch_size,model,X_train,y_train,X_test,y_test,True)
        start_punt+=batch_size
        y_trains.append(y_train)
        y_tests.append(y_test)
        train_preds.append(train_pred)
        test_preds.append(test_pred)
        epoch_hists.append(history)

    train_pred_tot = np.concatenate(train_preds,axis=0)
    test_pred_tot = np.concatenate(test_preds,axis=0)
    y_train_tot = np.concatenate(y_trains,axis=0)
    y_test_tot = np.concatenate(y_tests,axis=0)
    combined_epoch_history = {}
    for key in epoch_hists[0].history.keys():
        combined_epoch_history[key] = np.concatenate([h.history[key] for h in epoch_hists], axis=0)
    all_histories.append(combined_epoch_history)
    #Combine histories of all epochs together
    combined_history = {}
    for key in all_histories[0].keys():
        combined_history[key] = np.concatenate([h[key] for h in all_histories], axis=0)


    #plot de resultaten; op dit moment werkt dit alleen voor de laatste epoch van de laatste batch
    plot_results(combined_history,y_train_tot,y_test_tot,train_pred_tot,test_pred_tot,ROOT_DIR)

    results_array = [y_train_tot,y_test_tot,train_pred_tot,test_pred_tot]
    history_array = [combined_history['accuracy'],combined_history['val_accuracy'],combined_history['loss'],combined_history['val_loss']]
    try:
        np.save(ROOT_DIR+"/metrics.npy",results_array)
        np.save(ROOT_DIR+"/history.npy",history_array)
    except FileNotFoundError:
        np.save(ROOT_DIR+"/metrics.npy",results_array)
        np.save(ROOT_DIR+"/history.npy",history_array)


#%%
    
print(history)