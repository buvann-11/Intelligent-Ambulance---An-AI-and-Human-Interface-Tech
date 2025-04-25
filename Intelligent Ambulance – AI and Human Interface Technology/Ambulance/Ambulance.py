from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import numpy as np
import pandas as pd 
from tkinter.filedialog import askopenfilename
import tkinter
import numpy as np
from tkinter import filedialog
import socket
import pickle
import time

main = tkinter.Tk()
main.title("Ambulance Application")
main.geometry("900x500")

def reportCondition():
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    for i in range(len(dataset)):
        data = dataset[i]
        arr = []
        for j in range(len(data)):
            arr.append(str(data[j]))
        arr = ','.join(arr)
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        client.connect(('localhost', 2222))
        features = []
        features.append("patientdata")
        features.append(arr)
        features = pickle.dumps(features)
        client.send(features)
        data = client.recv(1000)
        data = data.decode()
        text.insert(END,"Patient Test Data = "+arr+" ===> "+data+"\n\n")
        text.update()
        text.update_idletasks()
        time.sleep(1)


font = ('times', 16, 'bold')
title = Label(main, text='Ambulance Application')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Report Patient Condition to Hospital Server", command=reportCondition)
upload.place(x=200,y=450)
upload.config(font=font1)  

font1 = ('times', 12, 'bold')
text=Text(main,height=18,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='turquoise')
main.mainloop()
