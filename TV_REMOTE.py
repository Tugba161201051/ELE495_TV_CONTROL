import cv2
import os
import tensorflow as tf
import pathlib
import numpy as np
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.protos import pipeline_pb2
from object_detection.utils import visualization_utils as viz_utils
from picamera import PiCamera
from picamera.array import PiRGBArray
import time
import io
from matplotlib import pyplot as plt
import vlc
import tf_record as xc


def give_command(command):           #tv komut numaraları  
    switcher = {
        1: "bir",
        2: "iki",
        3: "uc",
        4: "dort",
        5: "bes",
        6: "alti",
        7: "yedi",
        8: "sekiz",
        9: "dokuz",
        10: "sifir",
        11: "kanalartir",
        12: "kanalazalt",
        13: "sesac",
        14: "seskapa",
        15: "ackapa"
        }
    return switcher.get(command)

def lets_change(command):
    switch_command = {
        1: "irsend SEND_ONCE Toshiba KEY_1",
        2: "irsend SEND_ONCE Toshiba KEY_2",
        3: "irsend SEND_ONCE Toshiba KEY_3",
        4: "irsend SEND_ONCE Toshiba KEY_4",
        5: "irsend SEND_ONCE Toshiba KEY_5",
        6: "irsend SEND_ONCE Toshiba KEY_6",
        7: "irsend SEND_ONCE Toshiba KEY_7",
        8: "irsend SEND_ONCE Toshiba KEY_8",
        9: "irsend SEND_ONCE Toshiba KEY_9",
        10: "irsend SEND_ONCE Toshiba KEY_0",
        11: "irsend SEND_ONCE Toshiba KEY_CHANNELUP",
        12: "irsend SEND_ONCE Toshiba KEY_CHANNELDOWN",
        13: "irsend SEND_ONCE Toshiba KEY_VOLUMEUP",
        14: "irsend SEND_ONCE Toshiba KEY_VOLUMEDOWN",
        15: "irsend SEND_ONCE Toshiba KEY_POWER"
        }
    return switch_command.get(command)
   

config = config_util.get_configs_from_pipeline_file('/home/pi/Desktop/tsfw/tffiles/modeller/modelimiz/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config')

configs = config_util.get_configs_from_pipeline_file('/home/pi/Desktop/tsfw/tffiles/modeller/modelimiz/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/home/pi/Desktop/tsfw/tffiles/modeller/modelimiz/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/', 'ckpt-13')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections_1 = detection_model.postprocess(prediction_dict, shapes) #bu satır
    return detections_1
   

category_index = label_map_util.create_category_index_from_labelmap('/home/pi/Desktop/tsfw/tffiles/annotations/label_map.pbtxt')

camera = PiCamera()

camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))
rawCapture.truncate(0)

first_tour=1
old_command=0
channel_digit=0     #sayılar için
changing_digit=0    #kanal,ses komutları için
tv_channel_command=[]

back=time.time()
time.sleep(0.1)
font=cv2.FONT_HERSHEY_SIMPLEX


for screen in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

    image_np = np.copy(screen.array)
    image_np.setflags(write=1)
    input_data=np.expand_dims(image_np, 0)
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
        for key, value in detections.items()}
    detections['num_detections'] = num_detections
   
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
   
    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=.5,
        agnostic_mode=False)
   
    truth_rate = detections['detection_scores'][0]  #düşük ve algılanmama durumlarını engelleme
    command = detections['detection_classes'][0]+1
    cv2.putText(image_np_with_detections, "{0:.2f}".format(int(command)),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)    
    resized=cv2.resize(image_np_with_detections, (640,480))
    cv2.imshow('object detection', resized)
     
    threshold1 = 0.75
    threshold2 = 0.6
   
    if channel_digit>0 or changing_digit==1:
        chnaging_digit=0
        audio.stop()
       
    now= time.time()    
    time_difference=(now-back)
   
    if truth_rate<threshold2 and time_difference>2.9 :
       
        if truth_rate>0.3:
            first_tour=1
       
        while len(tv_channel_command) > 0 :
            print(tv_channel_command)
            print("basamak: ",channel_digit)
            print(tv_channel_command[0])
            os.system(tv_channel_command[0])
            time.sleep(0.2)
            del tv_channel_command[0]
        tv_channel_command=[] #tamponu boşalt
        channel_digit=0
       
    elif (truth_rate > threshold1) and time_difference>1.5:
        tv_channel=give_command(command)
        print(tv_channel)
       
        if command not in [11,12,13,14,15]:   #rakamlarsa hafızala, rakam olmayanları ele
            audio=vlc.MediaPlayer("/home/pi/Desktop/tsfw/audios/{0}.mp3".format(tv_channel))
            audio.play()
            tv_channel_command.append(lets_change(command))
            channel_digit+=1
            back=time.time()
           
           
        elif tv_channel_command== [] :  #hafıza boşken kanal-ses komutları gelirse
            print(command)
            audio=vlc.MediaPlayer("/home/pi/Desktop/tsfw/audios/{0}.mp3".format(tv_channel))
            audio.play()
            changing_digit=1
            os.system(lets_change(command))    

       
    elif truth_rate < threshold1 and truth_rate > threshold2 and time_difference>1.5 :
        #print('kücük')
       
        if(first_tour == 0):        #ilk kontrol değilse
            if(command == old_command)and command!=0 :   #hafızayla karşılaştır
               
                old_command=0   #sıfırla- bu tur hafızaya atma
                tv_channel=give_command(command)
                print('hafizali', old_truth, truth_rate, tv_channel)
                if command not in [11,12,13,14,15]:                    
                    audio=vlc.MediaPlayer("/home/pi/Desktop/tsfw/audios/{0}.mp3".format(tv_channel))
                    audio.play()
                    tv_channel_command.append(lets_change(command))
                   
                    channel_digit+=1
                    back=time.time()
                   
                elif tv_channel_command==[]  :              
                        print(command)
                        audio=vlc.MediaPlayer("/home/pi/Desktop/tsfw/audios/{0}.mp3".format(tv_channel))
                        audio.play()
                        changing_digit=1
                        os.system(lets_change(command))
                   
               
            else:   #2 düşük doğruluk oranı olan eşleşmiyorsa ele
                first_tour=1   #yeniden hafıza oluştur
     
        if(first_tour == 1):
            old_command = command
            old_truth = truth_rate
            first_tour=0       #hafızada tut
           
   
       
    key = cv2.waitKey(1) & 0xFF
     
    rawCapture.truncate(0)
   
    if key == ord("q"):        
        break
