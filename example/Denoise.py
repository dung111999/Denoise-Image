import cv2
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter.ttk import Style
import os
import tkinter as tk
import tkinter.ttk as exTk
import numpy as np
from PIL import Image,ImageTk,ImageDraw,ImageFont
import tkinter
from _overlapped import NULL
from pip._internal.utils import filetypes
import tkinter.messagebox as mbox
from sklearn.externals import joblib 
import imutils
import math
from tkinter import messagebox
class Example(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.bind("<Control-l>",lambda x: self.hide())
        self.initUI()

    def initUI(self):
        self.p = joblib.load("iris_knn3.pkl")
        print("LoadXong")
        self.parent.title("Denoise")
        self.undoredo = []
        self.imhientai = None

        menuBar = Menu(self.parent)
        self.parent.config(menu=menuBar)
        self.countredo = 1
        # Menu File
        fileMenu = Menu(menuBar)
        fileMenu.add_command(label="Add File",command=self.browseFile)
        fileMenu.add_command(label="Save as", command=self.saveImage)
        fileMenu.add_command(label="Exit", command=self.onExit)
        menuBar.add_cascade(label="File", menu=fileMenu)

        fileMenu2 = Menu(menuBar)
        fileMenu2.add_command(label="About",command=self.about)
        fileMenu2.add_command(label="Help",command=self.help)
        menuBar.add_cascade(label="Help", menu=fileMenu2)

        pic = ImageTk.PhotoImage(Image.open("newFile.png"))
        button1 = Button(text='New File',image=pic,compound=TOP,command = self.browseFile)
        button1.image = pic
        button1.grid(column=0, row=0)

        save = ImageTk.PhotoImage(Image.open("savee.png"))
        button1 = Button(text='Save', image=save, compound=TOP,command = self.saveImage)
        button1.image = save
        button1.grid(column=1, row=0)

        time = ImageTk.PhotoImage(Image.open("time.png"))
        button1 = Button(text='Original', image=time, compound=TOP, command = self.original)
        button1.image = time
        button1.grid(column=2, row=0)

        tentrai = ImageTk.PhotoImage(Image.open("tenphai.png"))
        button1 = Button(text='Undo', image=tentrai, compound=TOP,command=self.undoX)
        button1.image = tentrai
        button1.grid(column=3, row=0)

        tenphai = ImageTk.PhotoImage(Image.open("tentrai.png"))
        button1 = Button(text='Redo', image=tenphai, compound=TOP,command=self.redoX)
        button1.image = tenphai
        button1.grid(column=4, row=0)
        denoise = ImageTk.PhotoImage(Image.open("denoise.png"))
        button1 = Button(text='Denoise', image=denoise, compound=TOP, command=self.denoiseX)
        button1.image = denoise
        button1.grid(column=5, row=0)

        delete = ImageTk.PhotoImage(Image.open("delete.png"))
        button1 = Button(text='Delete', image=delete, compound=TOP,command=self.deleteX)
        button1.image = delete
        button1.grid(column=6, row=0)

        scrW = self.winfo_screenwidth()
        scrH = self.winfo_screenheight()
        self.frame1 = Frame(border=1, width=scrW-320, height=scrH-100)
        self.frame1.place(x=10,y=75)

        ##################
        scrW1 = scrW-320
        scrH1 = scrH-100
        self.frame2 = Frame(border=1,width=(scrW1/2)-200,height=(scrH1/2)-100)
        self.frame2.place(x=(scrW1/2)-250,y=(scrH1/2)-100)
        self.frame2.DAD_lb = Label(self.frame2,text="Drag and Drop",font= 'Times 27',fg='red')
        self.frame2.DAD_lb.place(x=150,y=0)

        self.frame2.YIH_lb = Label(self.frame2,text="Your images here", font='Times 17',fg='blue')
        self.frame2.YIH_lb.place(x=150,y=50)

        pic = ImageTk.PhotoImage(Image.open("file1.png"))
        self.file1_bt = Button(self.frame2,text='  Add File', image=pic, compound=LEFT,bg='yellow',fg='red',command = self.browseFile)
        self.file1_bt.image = pic
        self.file1_bt.place(x=150, y=100,width=100,height=30)

        pic = ImageTk.PhotoImage(Image.open("file2.png"))
        self.frame2.file2_lb = Label(self.frame2,image=pic)
        self.frame2.file2_lb.image = pic
        self.frame2.file2_lb.place(x=0, y=0)


        self.frame = Frame(border=1, width=300, height=650,bg='white')
        self.frame.place(relx=1,x=-8, y=100 , anchor=NE)
        label3= Label(self.frame,text='Toolbox',fg='red',bg='white')
        label3.place(anchor=NW)
        #label3.pack()

        self.button10 = Button(self.frame, text='X', fg='red',font='Times 10', bg='white',command=self.denoiseY)
        self.button10.place(relx=1,anchor=NE)

        denoise = ImageTk.PhotoImage(Image.open("denoise.png"))
        label1 = Label(self.frame,text='Denoise', image=denoise, compound=LEFT,bg='white')
        label1.image = denoise
        label1.place(x=5,y=35)
       
        #Do sang
        label3 = Label(self.frame, text='Brightness', fg='green', bg='white')
        label3.place(x=10,y=135,anchor=NW)
        #label3.pack()
        self.varBr = DoubleVar()
        scale = Scale(self.frame ,from_ = -255,to=255,variable = self.varBr,orient=HORIZONTAL,bg='white',fg='red')
        scale.place(x=10,y=170,width=270)
        #scale.pack()
        
        #Do tuong phan
        label3 = Label(self.frame, text='Contrast', fg='green', bg='white')
        label3.place(x=10, y=230, anchor=NW)
        #label3.pack()
        self.varCo = DoubleVar()
        scale = Scale(self.frame, from_=-40, to=40,variable = self.varCo,orient=HORIZONTAL, bg='white', fg='red')
        scale.place(x=10, y=255, width=270)
        #scale.pack()
        
        label3 = Label(self.frame,text='Gaussian',fg='green',bg='white')
        label3.place(x=10,y=425,anchor=NW)
        self.varhf = DoubleVar()
        scale = Scale(self.frame ,from_ = 0,to=20,variable = self.varhf,orient=HORIZONTAL,bg='white',fg='red')
        scale.place(x=10,y=455,width=270)
        
        label3 = Label(self.frame,text='Median',fg='green',bg='white')
        label3.place(x=10,y=335,anchor=NW)
        self.varhm = DoubleVar()
        scale2 = Scale(self.frame ,from_ = 0,to=20,variable = self.varhm,orient=HORIZONTAL,bg='white',fg='red')
        scale2.place(x=10,y=355,width=270)

        choose = ImageTk.PhotoImage(Image.open("choose.png"))
        button1 = Button(self.frame,text='Denoise', image=choose, compound=LEFT,bg='yellow',fg='red',command = self.denoiseColored)
        button1.image = choose
        button1.place(x=117,y=500)
        #button1.pack()
        
        button2 = Button(self.frame,text = "Run", compound = LEFT, bg = 'yellow', fg = 'red',command = self.getValue)
        button2.place(x=150,y=320)
        #button2.pack()

    def browseFile(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select A File", filetype=
        (("JPG File","*.jpg"),("JPEG File","*.jpeg"),("PNG File","*.png"),("All Files","*.*")))
        self.img = Image.open(self.filename)
        pic=ImageTk.PhotoImage(Image.open(self.filename))
        self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
        self.frame1.anh_hien_lb.image = pic
        w, h = self.img.size
        scrW = self.winfo_screenwidth()
        scrH = self.winfo_screenheight()
        scrW1=scrW - 320
        scrH1=scrH-100
        self.undoredo.append(cv2.imread(self.filename))
        self.countredo = 1
        self.imhientai = self.undoredo[len(self.undoredo)-1]
        self.frame1.anh_hien_lb.place(x=(scrW1/2)-w/2, y=(scrH1/2)-h/2-50, width=w + 50, height=h)
        self.frame2.destroy()
    def saveImage(self):
        im = self.img
        #caption = simpledialog.askstring("Label", "What would you like the label on your picture to say?")
        fontsize = 30
        if im.mode != "RGBA":
            im = im.convert("RGBA")
        txt = Image.new('RGBA', im.size, (255,255,255,0))

        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("arial.ttf", fontsize)
        draw.text((0, 0),"",(255,0,0),font=font)

        file = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=(("PNG file", "*.png"),("All Files", "*.*") ))
        if file:
            abs_path = os.path.abspath(file.name)
            out = Image.alpha_composite(im, txt)
            out.save(abs_path) # saves the image to the input file name
            
    def original(self):
        self.img = Image.open(self.filename)
        pic=ImageTk.PhotoImage(Image.open(self.filename))
        self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
        self.frame1.anh_hien_lb.image = pic
        w, h = self.img.size
        scrW = self.winfo_screenwidth()
        scrH = self.winfo_screenheight()
        scrW1 = scrW - 320
        scrH1 = scrH - 100
        self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)
        #self.frame1.anh_hien_lb.place(x=200, y=0, width=w+50, height=h)

        
    def extract_color_histogram(self,image, bins=(8, 8, 8)):
        # extract a 3D color histogram from the HSV color space using
        # the supplied number of `bins` per channel
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
            [0, 180, 0, 256, 0, 256])
     
        # handle normalizing the histogram if we are using OpenCV 2.4.X
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
     
        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
        else:
            cv2.normalize(hist, hist)
     
        # return the flattened histogram as the feature vector
        return hist.flatten()
    
    def sliding_window(self,img, step, windowSide):
        x = img.shape[1]
        y = img.shape[0]
        cut = []
        count = 0
        for i in range(0,x,step):
            for j in range(0,y,step):
                if j+windowSide<y and i+windowSide<x:
                    a = np.array(img[j:j+windowSide,i:i+windowSide])     
                    cut.append(a)
                    count +=1
        return cut   
    
    #Khu nhieu anh mau
    #Phuong phap khu nhieu khong cuc bo(Non-Local Means Denoising)
    def denoiseColored(self):
        img = self.imhientai
        hf=self.varhf.get()
        hf = int(hf)
        hm=self.varhm.get()
        hm = int(hm)
        if hm % 2 == 0 and hm != 0:
            hm += 1
        if hf % 2 == 0 and hm != 0:
            hf += 1
        dst = img

        if hm == 0 and hf == 0 :
            self.imhientai = self.imhientai
        elif hm != 0 and hf == 0:
            dst = cv2.medianBlur(img, hm)
        elif hm == 0 and hf != 0:
            dst = cv2.GaussianBlur(img, (hf, hf), 0)
        elif hm != 0 and hf != 0:
            dst = cv2.GaussianBlur(img, (hf, hf), 0)
            dst = cv2.medianBlur(img, hm)

        cv2.imwrite('anh1.jpg', dst)

        self.undoredo.append(dst)
        self.imhientai = dst
        self.img = Image.open('anh1.jpg')
        pic = ImageTk.PhotoImage(Image.open('anh1.jpg'))
        self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
        self.frame1.anh_hien_lb.image = pic
        w, h = self.img.size
        scrW = self.winfo_screenwidth()
        scrH = self.winfo_screenheight()
        scrW1 = scrW - 320
        scrH1 = scrH - 100
        self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)
        #Show anh ra
        
        danhgia = []
        scale = math.floor(min(dst.shape[1],dst.shape[0])/8)
        
        a = self.sliding_window(dst, scale , scale)
        for i in a:
            imgg = self.extract_color_histogram(i)
            imgg = imgg.reshape(1,-1)
            b = self.p.predict(imgg)
            danhgia.append(b)
        print(danhgia)
            
#         dg = (danhgia.count('0')/len(danhgia))
#         print("Muc do khu nhieu:" + str(dg)*100 + "%")
        nhieu = (danhgia.count('1') + danhgia.count('2'))
        all = len(danhgia)
        print("Muc do nhieu: " + str(nhieu/all) + "%")
        phantram = (nhieu/all) * 100
        messagebox.showinfo("Danh gia","Muc do nhieu "+str(phantram)+" %")
#         cv2.imshow("dst", dst)
#         cv2.waitKey()
#         cv2.destroyAllWindows()
        

    #Ham chinh sang va tuong phan   
    def getValue(self):
        img = self.imhientai
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        value = self.varBr.get()
        lim = 255 - value 
        v[v > lim] = 255
        v[v <= lim] = v[v <= lim] + value

        final_hsv = cv2.merge((h, s, v))
        img2 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
        if(self.varCo.get() != 0):
            clahe = cv2.createCLAHE(clipLimit=self.varCo.get(), tileGridSize=(8,8))
            lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
            l, a, b = cv2.split(lab)  # split on 3 different channels
            
            l2 = clahe.apply(l)  # apply CLAHE to the L-channel
            
            lab = cv2.merge((l2,a,b))  # merge channels
            img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
        cv2.imwrite('anh1.jpg', img2)

        self.undoredo.append(img2)
        self.imhientai = img2

        self.img = Image.open('anh1.jpg')
        pic = ImageTk.PhotoImage(Image.open('anh1.jpg'))
        self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
        self.frame1.anh_hien_lb.image = pic
        w, h = self.img.size
        scrW = self.winfo_screenwidth()
        scrH = self.winfo_screenheight()
        scrW1 = scrW - 320
        scrH1 = scrH - 100
        self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)

    #def onExit(self):
     #   self.quit()
    def onExit(self):
        self.quit()

    def denoiseX(self):
        self.frame.lift()

    def denoiseY(self):
        self.frame.lower()

    def undoX(self):
        dodai = len(self.undoredo)
        print(dodai)
        if self.countredo < dodai:
            print(self.countredo)

            self.img = self.undoredo[dodai - 1 - self.countredo]
            self.imhientai = self.img
            cv2.imwrite('anh1.jpg', self.img)
            pic = ImageTk.PhotoImage(Image.open('anh1.jpg'))
            self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
            self.frame1.anh_hien_lb.image = pic
            h, w, _ = self.img.shape
            scrW = self.winfo_screenwidth()
            scrH = self.winfo_screenheight()
            scrW1 = scrW - 320
            scrH1 = scrH - 100
            self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)
            self.countredo += 1
        elif self.countredo == dodai - 1:
            self.img = self.undoredo[0]
            self.imhientai = self.img
            cv2.imwrite('anh1.jpg', self.img)
            pic = ImageTk.PhotoImage(Image.open('anh1.jpg'))
            self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
            self.frame1.anh_hien_lb.image = pic
            h, w, _ = self.img.shape          
            scrW = self.winfo_screenwidth()
            scrH = self.winfo_screenheight()
            scrW1 = scrW - 320
            scrH1 = scrH - 100
            self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)

    def redoX(self):
        dodai = len(self.undoredo)
        print(dodai)
        if self.countredo > 0:
            self.countredo -= 1
            print(self.countredo)
            self.img = self.undoredo[dodai - 1 - self.countredo]
            self.imhientai = self.img
            cv2.imwrite('anh1.jpg', self.img)
            pic = ImageTk.PhotoImage(Image.open('anh1.jpg'))
            self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
            self.frame1.anh_hien_lb.image = pic
            h, w, _ = self.img.shape
            scrW = self.winfo_screenwidth()
            scrH = self.winfo_screenheight()
            scrW1 = scrW - 320
            scrH1 = scrH - 100
            self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)
        elif self.countredo == 0:
            self.img = self.undoredo[dodai - 1]
            self.imhientai = self.img
            cv2.imwrite('anh1.jpg', self.img)
            pic = ImageTk.PhotoImage(Image.open('anh1.jpg'))
            self.frame1.anh_hien_lb = Label(self.frame1, image=pic)
            self.frame1.anh_hien_lb.image = pic
            h, w, _ = self.img.shape
            scrW = self.winfo_screenwidth()
            scrH = self.winfo_screenheight()
            scrW1 = scrW - 320
            scrH1 = scrH - 100
            self.frame1.anh_hien_lb.place(x=(scrW1 / 2) - w / 2, y=(scrH1 / 2) - h / 2 - 50, width=w + 50, height=h)

    def deleteX(self):
        answer = messagebox.askquestion("Thông báo","Bạn có muốn xóa ảnh không")
        if answer == "yes" :
            self.undoredo = []
            scrW = self.winfo_screenwidth()
            scrH = self.winfo_screenheight()
            self.frame1 = Frame(border=1, width=scrW - 320, height=scrH - 100)
            self.frame1.place(x=10, y=75)

            scrW1 = scrW - 320
            scrH1 = scrH - 100
            self.frame2 = Frame(border=1, width=(scrW1 / 2) - 200, height=(scrH1 / 2) - 100)
            self.frame2.place(x=(scrW1 / 2) - 250, y=(scrH1 / 2) - 100)
            self.frame2.DAD_lb = Label(self.frame2, text="Drag and Drop", font='Times 27', fg='red')
            self.frame2.DAD_lb.place(x=150, y=0)

            self.frame2.YIH_lb = Label(self.frame2, text="Your images here", font='Times 17', fg='blue')
            self.frame2.YIH_lb.place(x=150, y=50)

            pic = ImageTk.PhotoImage(Image.open("file1.png"))
            self.file1_bt = Button(self.frame2, text='  Add File', image=pic, compound=LEFT, bg='yellow', fg='red',
                                   command=self.browseFile)
            self.file1_bt.image = pic
            self.file1_bt.place(x=150, y=100, width=100, height=30)

            pic = ImageTk.PhotoImage(Image.open("file2.png"))
            self.frame2.file2_lb = Label(self.frame2, image=pic)
            self.frame2.file2_lb.image = pic
            self.frame2.file2_lb.place(x=0, y=0)

    def about(self):
        messagebox.showinfo("Thông báo","Nhóm bao gồm các thành viên : "
                                                 "Phạm Tấn Dũng , "
                                                 "Trần Như Sơn và "
                                                 "Nguyễn Tử Hoàng Minh")
    def help(self):
        messagebox.showinfo("Hướng dẫn", "Các bạn chỉ cần add File và thực hiện các chức năng khử nhiễu , chỉnh sáng , tương phản")
root = Tk()
scrollbar = Scrollbar(root)
root.geometry("1000x1000+700+100")
app = Example(root)
root.mainloop()