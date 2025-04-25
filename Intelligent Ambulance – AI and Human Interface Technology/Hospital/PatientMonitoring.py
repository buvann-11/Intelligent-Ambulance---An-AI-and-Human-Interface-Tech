from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import pickle



main = tkinter.Tk()
main.title("Intelligent Ambulance – AI and Human Interface Technology")
main.geometry("1300x1200")


global filename
global dataset
accuracy = []
precision = []
recall = []
fscore = []
global  X_train, X_test, y_train, y_test
global classifier
global X, Y, sc
labels = ['Normal Heart Rate', 'Abnormal Heart Rate'] 

def upload():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset.head())+"\n")
    text.insert(END,"Dataset contains total records    : "+str(dataset.shape[0])+"\n")
    text.insert(END,"Dataset contains total attributes : "+str(dataset.shape[1])+"\n")
    label = dataset.groupby('target').size()
    label.plot(kind="bar")
    plt.show()

def processDataset():
    global X_train, X_test, y_train, y_test, sc
    global X, Y
    global dataset
    text.delete('1.0', END)
    data = dataset.values
    X = data[:,0:data.shape[1]-1]
    Y = data[:,data.shape[1]-1]
    Y = Y.astype(int)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) #shuffling the dataset
    X = X[indices]
    Y = Y[indices]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    text.insert(END,"Dataset Preprocessing, Normalizing & Shuffling Task Completed\n")
    text.insert(END,str(X)+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    text.insert(END,"Dataset train and test split details\n\n")
    text.insert(END,"80% records used for Ensemble training Algorithm : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% records used for Ensemble testing Algorithm  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict):
    global labels
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show() 

def runDecisionTree():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    global X_train, X_test, y_train, y_test

    dt_cls = DecisionTreeClassifier() 
    dt_cls.fit(X_train, y_train)
    predict = dt_cls.predict(X_test)
    calculateMetrics("Decision Tree", y_test,predict)

def runRandomForest():
    global X_train, X_test, y_train, y_test, classifier
    rf_cls = RandomForestClassifier() 
    rf_cls.fit(X_train, y_train)
    classifier = rf_cls
    predict = rf_cls.predict(X_test)
    calculateMetrics("Random Forest", y_test, predict)
    
def runKNN():
    global X_train, X_test, y_train, y_test, classifier
    knn_cls = KNeighborsClassifier(n_neighbors = 10) 
    knn_cls.fit(X_train, y_train)
    predict = knn_cls.predict(X_test)
    calculateMetrics("KNN", y_test, predict)    


def graph():
    df = pd.DataFrame([['Decision Tree','Precision',precision[0]],['Decision Tree','Recall',recall[0]],['Decision Tree','F1 Score',fscore[0]],['Decision Tree','Accuracy',accuracy[0]],
                       ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                       ['KNN','Precision',precision[2]],['KNN','Recall',recall[2]],['KNN','F1 Score',fscore[2]],['KNN','Accuracy',accuracy[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

running = True

def startCloudServer():
    global text
    class CloudThread(Thread):  
        def __init__(self,ip,port): 
            Thread.__init__(self)
            global text
            self.ip = ip 
            self.port = port 
            print('Request received from Ambulance IP : '+ip+' with port no : '+str(port)+"\n") 
 
        def run(self): 
            data = conn.recv(1000)
            dataset = pickle.loads(data)
            request = dataset[0]
            if request == "patientdata":
                data = dataset[1]
                text.delete('1.0', END)
                text.insert(END,"Patient Data Received : "+str(data)+"\n")
                text.update_idletasks()
                output = predict(data)
                conn.send(output.encode())
                text.insert(END,output+"\n\n")               
            
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 2222))
    threads = []
    text.insert(END,"Cloud Server Started\n\n")
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = CloudThread(ip,port) 
        newthread.start() 
        threads.append(newthread) 
    for t in threads:
        t.join()

def startServer():
    Thread(target=startCloudServer).start()    

def predict(data):
    data = data.split(",")
    temp = []
    for i in range(len(data)):
        temp.append(float(data[i]))
    return predictCondition(temp)    

def predictCondition(testData):
    global sc, classifier
    data = []
    data.append(testData)
    data = np.asarray(data)
    data = sc.transform(data)
    predict = classifier.predict(data)
    predict = predict[0]
    msg = "Predicted Output: Patient Condition Normal"
    if predict == 1:
        msg = "Predicted Output: Patient Condition Abnormal"
    return msg

font = ('times', 16, 'bold')
title = Label(main, text='Intelligent Ambulance – AI and Human Interface Technology')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Heart Disease Dataset", command=upload)
upload.place(x=890,y=100)
upload.config(font=font1)  

processButton = Button(main, text="Dataset Preprocessing & Train Test Split", command=processDataset)
processButton.place(x=890,y=150)
processButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=890,y=200)
dtButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
rfButton.place(x=890,y=250)
rfButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=890,y=300)
knnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=890,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Receive Patient Condition to Hospital Server", command=startServer)
predictButton.place(x=890,y=400)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
