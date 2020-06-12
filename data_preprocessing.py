#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:20 2020

@author: Yan
"""

import os
import zipfile
import cv2
import dlib
from util import out_put
from PIL import Image
from io import BytesIO

        
def face(img, alignment = False):
    
    '''
    Can be used to detect,Crop and Normalize the face. 
    Assume the image only contains one single frontal face
   
    Parameters
    ----------
    img       : array, shape (height,width,3), the img in RGB representation which contains the face
    alignment : boolean, default False, whether to normalize the face
    '''
    
    rect = None
    
    #santiy check, if the detector can not find the faces, return None
    try:
        rect = dlib_detector(img, 0)[0]
    except IndexError:
        return None
    
    if alignment:
        
        return img[rect.top():rect.bottom(),rect.left():rect.right()]
    
    faces = dlib.full_object_detections()
    faces.append(predictor(img, rect))
    img = dlib.get_face_chip(img, faces[0],224)
    return img



def write(flag,path,process,file_name,outputZipName):        
    '''
    This is a help function. It extract frames and processes frames
    Parameters
    ----------
    flag                   : boolean, create a new zip or append to the zip
    path                   : str, the path to read the video files
    file_name              : str, the name of folder to store the processed data. 
    outputZipname          : str, the path and names to store the preprocessed data
    process                : int, default 0,  the level of preprocessing wants.
                                  0 indicates, no preprocessing will used, only extract raw frames,
                                  1 indicates, only crop the faces,
                                  2 indicates, crop and normalize the face    
    '''
    output_zip = zipfile.ZipFile(f"{outputZipName}.zip",mode = flag and "w" or "a")
    cap = cv2.VideoCapture(path)
    rec,frame = cap.read()
    
    c = 0
    while rec:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        if process == 1:
            img = face(frame)
        elif process == 2:
            img = face(frame,True)
        
        if img is not None:
            assert frame.shape[-1] == 3
            img_file = BytesIO()
            Image.fromarray(img).save(img_file,"JPEG")
            name = os.path.join(f"{file_name}",f"{c}.jpg")
            output_zip.writestr(name,img_file.getvalue())
        
        rec,frame = cap.read()
        
        c+=1
        
    output_zip.close()  


def extract_frames(
        path,
        outputZipName,
        pretrained_modelpath = None,
        verbose  = "processes",
        process  = 0,
        database = None
        ):
    
    '''
    The function used to preprocesses the datas. 
    
    The structure of the UVA-NEMO databases is like
    .../videos
        -- xxx.mp4
        -- xxx.mp4
        .
        .
        .
        -- xxx.mp4
        
    The structure of BBC database is like
    
    .../Genuine
        -- xxx.mp4
        -- xxx.mp4
        .
        .
        .
        -- xxx.mp4
        
    .../posed
        -- xxx.mp4
        -- xxx.mp4
        .
        .
        .
        -- xxx.mp4
        
    The structure of MMI database is like
    
    .../Sessions
       .../x
           -- xxxx.avi
           -- xxxx.xml
           -- session.xml
       .../x
           -- xxxx.avi
           -- xxxx.xml
           -- session.xml
       .
       .
       .
       .../x
           -- xxxx.avi
           -- xxxx.xml
           -- session.xml           
          
    The SPOS database provided processed data, thus no-preprocessing is required.    
    Parameters
    ----------
    path                   : str, the path to UVA-NEMO database video folders
    pretrained_modelpath   : str, the path that contains the pretrained model to do face detection,normalization, etc
    outputZipname          : str, the path and names to store the preprocessed data
    verbose                : str, default processes, the path and names to output the schedule of data preprocessing,
    process                : int, default 0,  the level of preprocessing wants.
                                  0 indicates, no preprocessing will used, only extract raw frames,
                                  1 indicates, only crop the faces,
                                  2 indicates, crop and normalize the face
    
    database               :str, the name of the databases to extract, the valide name is
                           "UVA-NEMO","BBC","MMI"
                                  
    '''
    
    global dlib_detector
    global predictor
    
    #santity check, if process = 0, then pretrained_modelpath != None.
    assert  process in [0,1,2], "Please provide valid process number"
    assert  process == 0 or pretrained_modelpath != None, "Please provide the pretrained models, if data preprocessing required."
    assert  database in ["UVA-NEMO","BBC","MMI"], "Invalid database name"
    
    cs = 0
    flag = True
    dlib_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(f"{pretrained_modelpath}")
    
    if database == "UVA-NEMO":
    
        for location,file in enumerate(sorted(os.listdir(path))):
            
            if not file.endswith(".mp4"):
                continue
            
            write(flag,os.path.join(path,file),process,file.replace(".mp4",""),outputZipName)
            flag = False
    
            
            if cs % 50 == 0 and verbose != None:
                out_put(f"processed {cs/1240} files",verbose)
                    
            cs+= 1
            
    elif database == "BBC":
        
        for file in os.listdir(path):
            p = os.path.join(path,file)
            if not os.path.isdir(p):
                continue
            
            for avi in os.listdir(p):
                
                if not avi.endswith(".mp4"):
                    continue
          
                write(flag,os.path.join(p,avi),process,file+avi,outputZipName)
                flag = False
                out_put(f"processed {cs} files",verbose)
                cs+= 1
                
    elif database == "MMI":
        
        for file in os.listdir(path):
            p = os.path.join(path,file)
            session_id = file
            if not os.path.isdir(p):
                continue
            
            avi_name = [i for i in os.listdir(p) if "session" not in i][0].split(".")[0]
        
            avi_name = os.path.join(p,avi_name + ".avi")
        
            write(flag,avi_name,process,session_id,outputZipName)
            flag = False
            out_put(f"processed {cs} files",verbose)
            cs+= 1

        
        
        
        
        
        