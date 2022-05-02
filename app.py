import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import queue
from tensorflow.keras.models import model_from_json
import glob

st.set_page_config(page_title="Sign Language Recognition", page_icon = "logo2.png", layout = "centered", initial_sidebar_state = "expanded") 
st.title("Sign Language Recognition")

if 'images' not in st.session_state:
    st.session_state.images = glob.glob('images/*.jpg')
    
with st.sidebar:
    st.title("ASL CHARACTERS")
    rows = [st.columns(3) for _ in range(9)]
    cols = [column for row in rows for column in row]
    for col, Image in zip(cols, st.session_state.images):
        col.image(Image)


@st.cache(allow_output_mutation=True)
def Load_Models():
    json_file = open("models/new-model-bw.json", "r")
    model_json = json_file.read()
    json_file.close()
    main_model = model_from_json(model_json)
    main_model.load_weights("models/new-model-bw.h5") 
    
    json_file_aemnst = open("models/model-bw-aemnst.json", "r")
    model_json_aemnst = json_file_aemnst.read()
    json_file_aemnst.close()
    model_aemnst = model_from_json(model_json_aemnst)
    model_aemnst.load_weights("models/model-bw-aemnst.h5") 
    
    json_file_dru = open("models/model-bw-dru.json", "r")
    model_json_dru = json_file_dru.read()
    json_file_dru.close()
    model_dru = model_from_json(model_json_dru)
    model_dru.load_weights("models/model-bw-dru.h5") 
    
    return (main_model, model_aemnst, model_dru)
        
main_model, model_aemnst, model_dru = Load_Models()

class VideoProcessor:
    def __init__(self):
      self.result_queue = queue.Queue()
      
    def aemnst(self, roi):
        preds = model_aemnst.predict(roi.reshape(1, 64, 64, 1))
        preds = np.argmax(preds, axis=1)
        
        if preds == 0:
            return "A"
        if preds == 1:
            return "E"
        if preds == 2:
            return "M"
        if preds == 3:
            return "N"
        if preds == 4:
            return "S"
        if preds == 5:
            return "T"
            
    def dru(self, roi):
        preds = model_dru.predict(roi.reshape(1, 64, 64, 1))
        preds = np.argmax(preds, axis=1)
        
        if preds == 0:
            return "D"
        if preds == 1:
            return "R"
        if preds == 2:
            return "U"
        
        
    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        img = cv2.flip(frame, 1)
             
        x1 = 350
        y1 = 10
        x2 = 606
        y2 = 266
                
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
               
        roi=img[y1:y2,x1:x2].copy()
        roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        save_img = cv2.resize(roi, (64,64))
                            
        preds = main_model.predict(save_img.reshape(1, 64, 64, 1))

        preds = np.argmax(preds, axis=1)

        if preds == 0:  #A
            output = self.aemnst(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds == 1:#B
            self.result_queue.put("B")
            cv2.putText(img,"B",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==2:  #c
            self.result_queue.put("C")
            cv2.putText(img,"C",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==3: #D
            output = self.dru(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==4: #E
            output = self.aemnst(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==5: #F
            #output = self.vwf(save_img)
            self.result_queue.put("F")
            cv2.putText(img,"F",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==6: #G
            self.result_queue.put("G")
            cv2.putText(img,"G",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==7: #H
            self.result_queue.put("H")
            cv2.putText(img,"H",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==8: #I
            self.result_queue.put("I")
            cv2.putText(img,"I",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==9: #J
            self.result_queue.put("J")
            cv2.putText(img,"J",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==10:#K
            self.result_queue.put("K")
            cv2.putText(img,"K",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==11:#L 
            self.result_queue.put("L")
            cv2.putText(img,"L",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==12:#M
            output = self.aemnst(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==13:#N
            output = self.aemnst(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==14:#O
            self.result_queue.put("O")
            cv2.putText(img,"O",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==15:#P
            self.result_queue.put("P")
            cv2.putText(img,"P",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==16:#Q
            self.result_queue.put("Q")
            cv2.putText(img,"Q",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==17:#R
            output = self.dru(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==18:#S
            output = self.aemnst(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==19:#T
            output = self.aemnst(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==20:#U
            output = self.dru(save_img)
            self.result_queue.put(output)
            cv2.putText(img,output,(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==21:#V
            #output = self.vwf(save_img)
            self.result_queue.put("V")
            cv2.putText(img,"V",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==22:#W
            #output = self.vwf(save_img)
            self.result_queue.put("W")
            cv2.putText(img,"W",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==23:#X
            self.result_queue.put("X")
            cv2.putText(img,"X",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==24:#Y
            self.result_queue.put("Y")
            cv2.putText(img,"Y",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        elif preds==25:#Z
            self.result_queue.put("Z")
            cv2.putText(img,"Z",(18,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        else:
            self.result_queue.put(" ")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

col1, col2, col3 = st.columns([6,0.5,2])

with col1:
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration={  
            "iceServers": [
                {"urls": ["stun:stun.services.mozilla.com"]},
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col3:
    st.write("Present Character")
    character_placeholder = st.empty()
    st.write("Word")
    word_placeholder = st.empty()

    if st.checkbox("Show the detected labels", value=True):
        if ctx.state.playing:       
            word = [0 for i in range(26)]
            count = 0
            word1 = ""
           
            while True:
                if ctx.video_processor:
                    try:
                        result = ctx.video_processor.result_queue.get(
                            timeout = 1.0
                        )
                    except queue.Empty:
                         result = None
                    
                    character_placeholder.write(result)

                    if count < 30:
                        if result != None and ord(result) != 32:
                                word[ord(result) - 65] += 1                    
                        count=count+1
                    else:
                        word1 = word1 + chr(65 + word.index(max(word)))
                        word_placeholder.write(word1)
                        count = 0
                        word = [0 for i in range(26)]
                else:
                    break
