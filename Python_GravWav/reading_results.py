import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

history={}
ROOT_DIR = r'C:\Users\JFK-9\Documents\_Studie\CA Machine Learning\Lokale_Output'
y_train_tot,y_test_tot,train_pred_tot,test_pred_tot=np.load(ROOT_DIR+"/metrics.npy")
history['accuracy'],history['val_accuracy'],history['loss'],history['val_loss']=np.load(ROOT_DIR+"/history.npy")


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
        plt.show()

    train_cm = confusion_matrix(y_train_b, train_pred_b)
    test_cm = confusion_matrix(y_test_b, test_pred_b)
    plot_confusion_matrix(train_cm, classes=range(2), title='Training Confusion Matrix',set="train")
    plot_confusion_matrix(test_cm, classes=range(2), title='Validation Confusion Matrix',set="test")

    print("Confusion Matrices plotted")
plot_results(history,y_train_tot,y_test_tot,train_pred_tot,test_pred_tot,ROOT_DIR)