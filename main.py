##############################################
# sudo apt-get install -y python3-picamera
# sudo -H pip3 install imutils --upgrade
##############################################

import sys
import numpy as np
import board
import adafruit_dht
import RPi.GPIO as GPIO
import cv2, io, time, argparse, re
from os import system
from os.path import isfile, join
from time import sleep
import multiprocessing as mp
try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin
import heapq
import threading
#import flask 
from flask import Flask, render_template, request
app = Flask(__name__)

try:
    from imutils.video.pivideostream import PiVideoStream
    from imutils.video.filevideostream import FileVideoStream
    import imutils
except:
    pass

lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
vs = None
window_name = ""
elapsedtime = 0.0

g_plugin = None
g_inferred_request = None
g_heap_request = None
g_inferred_cnt = 0
g_number_of_allocated_ncs = 0

LABELS = ["neutral", "happy", "sad", "surprise", "anger"]
COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))
recommended_drink = ""
temperature = 0
humidity = 0


def camThread(LABELS, resultsEm, frameBuffer, camera_width, camera_height, vidfps, number_of_camera, mode_of_camera):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global vs
    global window_name


    if mode_of_camera == 0:
        cam = cv2.VideoCapture(number_of_camera)
        if cam.isOpened() != True:
            print("USB Camera Open Error!!!")
            sys.exit(0)
        cam.set(cv2.CAP_PROP_FPS, vidfps)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        window_name = "USB Camera"
    else:
        vs = PiVideoStream((camera_width, camera_height), vidfps).start()
        sleep(3)
        window_name = "PiCamera"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        t1 = time.perf_counter()

        # USB Camera Stream or PiCamera Stream Read
        color_image = None
        if mode_of_camera == 0:
            s, color_image = cam.read()
            if not s:
                continue
        else:
            color_image = vs.read()

        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        height = color_image.shape[0]
        width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        res = None

        if not resultsEm.empty():
            res = resultsEm.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults)

        cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))

        if cv2.waitKey(1)&0xFF == ord('q'):
            sys.exit(0)

        ## Print FPS
        framecount += 1
        if framecount >= 25:
            fps       = "(Playback) {:.1f} FPS".format(time1/25)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount/time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2-t1
        time1 += 1/elapsedTime
        time2 += elapsedTime


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue
## here~~~~~~
## here~~~~~~~
def async_infer(ncsworkerFd, ncsworkerEm):
    ##initialize the time value
    start_time = int(time.perf_counter())
    end_time = int(time.perf_counter())
    print("1.start:"+str(start_time)+" end:"+str(end_time))

    while end_time - start_time <= 2:
        ncsworkerFd.predict_async()
        ncsworkerEm.predict_async()
        end_time = int(time.perf_counter())
    
    final_emotions = ncsworkerEm.count_emotion()
    print(final_emotions)      
    return final_emotions
    #processes.terminate();
    
class BaseNcsWorker():

    def __init__(self, devid, model_path, number_of_ncs):
        global g_plugin
        global g_inferred_request
        global g_heap_request
        global g_inferred_cnt
        global g_number_of_allocated_ncs

        self.devid = devid
        if number_of_ncs   == 0:
            self.num_requests = 4
        elif number_of_ncs == 1:
            self.num_requests = 4
        elif number_of_ncs == 2:
            self.num_requests = 2
        elif number_of_ncs >= 3:
            self.num_requests = 1

        print("g_number_of_allocated_ncs =", g_number_of_allocated_ncs, "number_of_ncs =", number_of_ncs)

        if g_number_of_allocated_ncs < 1:
            self.plugin = IEPlugin(device="MYRIAD")
            self.inferred_request = [0] * self.num_requests
            self.heap_request = []
            self.inferred_cnt = 0
            g_plugin = self.plugin
            g_inferred_request = self.inferred_request
            g_heap_request = self.heap_request
            g_inferred_cnt = self.inferred_cnt
            g_number_of_allocated_ncs += 1
        else:
            self.plugin = g_plugin
            self.inferred_request = g_inferred_request
            self.heap_request = g_heap_request
            self.inferred_cnt = g_inferred_cnt

        self.model_xml = model_path + ".xml"
        self.model_bin = model_path + ".bin"
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)


class NcsWorkerFd(BaseNcsWorker):

    def __init__(self, devid, frameBuffer, resultsFd, model_path, number_of_ncs):

        super().__init__(devid, model_path, number_of_ncs)
        self.frameBuffer = frameBuffer
        self.resultsFd   = resultsFd


    def image_preprocessing(self, color_image):

        prepimg = cv2.resize(color_image, (300, 300))
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            color_image = self.frameBuffer.get()
            prepimg = self.image_preprocessing(color_image)
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1

                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0

                self.exec_net.requests[reqnum].wait(-1)
                out = self.exec_net.requests[reqnum].outputs["detection_out"].flatten()

                detection_list = []
                face_image_list = []

                for detection in out.reshape(-1, 7):

                    confidence = float(detection[2])

                    if confidence > 0.3:
                        detection[3] = int(detection[3] * color_image.shape[1])
                        detection[4] = int(detection[4] * color_image.shape[0])
                        detection[5] = int(detection[5] * color_image.shape[1])
                        detection[6] = int(detection[6] * color_image.shape[0])
                        if (detection[6] - detection[4]) > 0 and (detection[5] - detection[3]) > 0:
                            detection_list.extend(detection)
                            face_image_list.extend([color_image[int(detection[4]):int(detection[6]), int(detection[3]):int(detection[5]), :]])

                if len(detection_list) > 0:
                    self.resultsFd.put([detection_list, face_image_list])

                self.inferred_request[reqnum] = 0


        except:
            import traceback
            traceback.print_exc()


class NcsWorkerEm(BaseNcsWorker):

    def __init__(self, devid, resultsFd, resultsEm, model_path, number_of_ncs):

        super().__init__(devid, model_path, number_of_ncs)
        self.resultsFd = resultsFd
        self.resultsEm = resultsEm
        self.resultsList = []
        self.emoDict = {'happy':0 , 'surprise':0 , 'sad':0 , 'anger':0, 'neutral':0 }

    def count_emotion(self):
        print(self.resultsList)
        for emo in self.resultsList:
            self.emoDict[emo] = self.emoDict[emo] +1
        
        max_value = max(self.emoDict.values())  # maximum value
        max_key = [k for k, v in self.emoDict.items() if v == max_value] # getting all keys containing the `maximum`
        
        return max_key
    def image_preprocessing(self, color_image):

        try:
            prepimg = cv2.resize(color_image, (64, 64))
        except:
            prepimg = np.full((64, 64, 3), 128)
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        return prepimg


    def predict_async(self):
        try:

            if self.resultsFd.empty():
                return

            resultFd = self.resultsFd.get()
            detection_list  = resultFd[0]
            face_image_list = resultFd[1]
            emotion_list    = []
            max_face_image_list_cnt = len(face_image_list)
            image_idx = 0
            end_cnt_processing = 0
            heapflg = False
            cnt = 0
            dev = 0

            if max_face_image_list_cnt <= 0:
                detection_list.extend([""])
                self.resultsEm.put([detection_list])
                return
            
            while True:
                
                reqnum = searchlist(self.inferred_request, 0)

                if reqnum > -1 and image_idx <= (max_face_image_list_cnt - 1) and len(face_image_list[image_idx]) > 0:

                    if len(face_image_list[image_idx]) == []:
                        image_idx += 1
                        continue
                    else:
                        prepimg = self.image_preprocessing(face_image_list[image_idx])
                        image_idx += 1

                    self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: prepimg})
                    self.inferred_request[reqnum] = 1
                    self.inferred_cnt += 1
                    if self.inferred_cnt == sys.maxsize:
                        self.inferred_request = [0] * self.num_requests
                        self.heap_request = []
                        self.inferred_cnt = 0
                    heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))
                    heapflg = True

                if heapflg:
                    cnt, dev = heapq.heappop(self.heap_request)
                    heapflg = False

                if self.exec_net.requests[dev].wait(0) == 0:
                    self.exec_net.requests[dev].wait(-1)
                    out = self.exec_net.requests[dev].outputs["prob_emotion"].flatten()
                    emotion = LABELS[int(np.argmax(out))]
                    ##print(emotion)
                    detection_list.extend([emotion])
                    # print(detection_list)
                    
                    self.resultsList.append(emotion)
                    self.resultsEm.put([detection_list])
                    self.inferred_request[dev] = 0
                    end_cnt_processing += 1
                    if end_cnt_processing >= max_face_image_list_cnt:
                        break
                    
                    return detection_list
                
                else:
                    heapq.heappush(self.heap_request, (cnt, dev))
                    heapflg = True
                
        except:
            import traceback
            traceback.print_exc()


def inferencer(resultsFd, resultsEm, frameBuffer, number_of_ncs, fd_model_path, em_model_path):

    # Init infer threads
    threads = []

    
    for devid in range(number_of_ncs):
        # Face Detection, Emotion Recognition start
        nc = NcsWorkerFd(devid, frameBuffer, resultsFd, fd_model_path, number_of_ncs)
        em = NcsWorkerEm(devid, resultsFd, resultsEm, em_model_path, 0)
        emotion = async_infer(nc,em)
        return emotion
        '''
        thworker = threading.Thread(target=async_infer, args=nc,em,)
        thworker.start()
        threads.append(thworker)
        print(thworker)
        print("Thread-"+str(devid))
        
    for th in threads:
        th.join()
        '''
    
    
    


def overlay_on_image(frames, object_infos):

    try:

        color_image = frames

        if isinstance(object_infos, type(None)):
            return color_image

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]
        entire_pixel = height * width
        img_cp = color_image.copy()

        for object_info in object_infos:

            if object_info[2] == 0.0:
                break

            if (not np.isfinite(object_info[0]) or
                not np.isfinite(object_info[1]) or
                not np.isfinite(object_info[2]) or
                not np.isfinite(object_info[3]) or
                not np.isfinite(object_info[4]) or
                not np.isfinite(object_info[5]) or
                not np.isfinite(object_info[6])):
                continue

            min_score_percent = 60
            source_image_width = width
            source_image_height = height
            percentage = int(object_info[2] * 100)

            if (percentage <= min_score_percent):
                continue

            box_left   = int(object_info[3])
            box_top    = int(object_info[4])
            box_right  = int(object_info[5])
            box_bottom = int(object_info[6])
            emotion    = str(object_info[7])

            label_text = emotion + " (" + str(percentage) + "%)"

            box_color =  COLORS[searchlist(LABELS, emotion, 0)]
            box_thickness = 2
            cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)
            label_background_color = (125, 175, 75)
            label_text_color = (255, 255, 255)
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_left = box_left
            label_top = box_top - label_size[1]
            if (label_top < 1):
                label_top = 1
            label_right = label_left + label_size[0]
            label_bottom = label_top + label_size[1]
            cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
            cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


        cv2.putText(img_cp, fps,       (width-170,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width-170,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        return img_cp

    except:
        import traceback
        traceback.print_exc()

def get_Environment_Data():
    dhtDevice = adafruit_dht.DHT22(board.D4)
    global temperature
    global humidity
    
    try:
        temperature = dhtDevice.temperature
        humidity = dhtDevice.humidity
        #print(temperature,humidity)
        print("Temp: {:.1f} C Humidity: {}%".format(temperature,humidity))
        return temperature,humidity
    
    except RuntimeError as error:
        # Errors happen fairly often, DHT's are hard to read, just keep going
        print(error.args[0])
    
    #get emotion data from main function 
def RecommendDrink(emotion):
    
    global recommended_drink 
    global temperature
    global humidity
    user_score = 0
    
    #20% weighted score
    if temperature <= 25:
        user_score-= 1*0.2
    else:
        user_score+= 1*0.2
    
    #20% weighted score
    if humidity <= 70:
        user_score-= 1*0.2
    else:
        user_score+= 1*0.2
    
    #60% weighted score    
    if emotion == "anger":
        user_score+= 1*0.6  
    elif emotion == "sad":    
        user_score+= 2*0.6 
    elif emotion == "neutral":
        user_score+= 3*0.6 
    elif emotion == "happy":
        user_score+= 4*0.6 
    elif emotion == "surprise":
        user_score+= 5*0.6
       
    print("user_score: "+str(user_score))
    
    if user_score >= 0.19 and user_score < 1:
        print("Recommend: Sprite_More, Score: "+str(user_score))
        recommended_drink = "Sprite_More"
        return recommended_drink
        
    elif user_score >= 1 and user_score < 1.8:
        print("Recommend: Sprite, Score: "+str(user_score))
        recommended_drink = "Sprite"
        return recommended_drink
    
    elif user_score >= 1.8 and user_score < 2.6:
        print("Recommend: Coke, Score: "+str(user_score))
        recommended_drink = "Coke"
        return recommended_drink
    
    elif user_score >= 2.6 and user_score < 3.4 :
        print("Recommend: Coke_More, Score: "+str(user_score))
        recommended_drink = "Coke_More"
        return recommended_drink
    
    else:
        print("there's something wrong! help...")

def makeDrink(drink):
    
    GPIO.setmode(GPIO.BCM)
    global recommended_drink
    #set recommended drink to drink variable
    drink = recommended_drink
    
    # pump 1 - Coke (pitender right side)
    GPIO.setup(20, GPIO.OUT, initial=GPIO.HIGH)
    # pump 2 - Sprite
    GPIO.setup(21, GPIO.OUT, initial=GPIO.HIGH)
    print("Done GPIO initializing")
    
    if drink == "Sprite_More":
        GPIO.output(20, GPIO.LOW)
        sleep(1.5)
        GPIO.output(20, GPIO.HIGH)
        
        GPIO.output(21, GPIO.LOW)
        sleep(3.5)
        GPIO.output(21, GPIO.HIGH)
        
    elif drink == "Sprite":
        GPIO.output(21, GPIO.LOW)
        sleep(5)
        GPIO.output(21, GPIO.HIGH)
        
    elif drink == "Coke":
        GPIO.output(20, GPIO.LOW)
        sleep(5)
        GPIO.output(20, GPIO.HIGH)
        
    elif drink == "Coke_More":
        GPIO.output(20, GPIO.LOW)
        sleep(3.5)
        GPIO.output(20, GPIO.HIGH)
        
        GPIO.output(21, GPIO.LOW)
        sleep(1.5)
        GPIO.output(21, GPIO.HIGH)
    
    print("drink done =)")
    GPIO.cleanup()

@app.route("/")
def start():
    return render_template('main.html')

@app.route("/control/")
def main():
    
    mode_of_camera = 1
    number_of_camera = 1
    camera_width  = 640
    camera_height = 480
    number_of_ncs = 1
    vidfps = 30
    fd_model_path = "./FP16/face-detection-retail-0004"
    em_model_path = "./FP16/emotions-recognition-retail-0003"
    
   # try:

    mp.set_start_method('forkserver')
    frameBuffer = mp.Queue(10)
    resultsFd = mp.Queue() # Face Detection Queue
    resultsEm = mp.Queue() # Emotion Recognition Queue
    
    
    # Start streaming
    p = mp.Process(target=camThread,
                   args=(LABELS, resultsEm, frameBuffer, camera_width, camera_height, vidfps, number_of_camera, mode_of_camera),
                   daemon=True)
    p.start()
    processes.append(p)

    
    #get emotion data   
    emotion_data = inferencer(resultsFd, resultsEm, frameBuffer, number_of_ncs, fd_model_path, em_model_path)
    emotion = ' '.join(emotion_data)
    
    recommend_drink = RecommendDrink(emotion)
    
    makeDrink(recommend_drink)
    
    # capacity: 600ml
    coke_capacity = 600
    sprite_capacity = 600
    
    # 1sec will output 25ml liquid
    if recommend_drink == "Sprite_More":
        coke_capacity-= 25*1.5
        sprite_capacity-= 25*3.5
    
    elif recommend_drink == "Sprite":
        sprite_capacity-= 25*5
        
    elif recommend_drink == "Coke":
        coke_capacity-= 25*5
        
    elif recommend_drink == "Coke_More":
        coke_capacity-= 25*3.5
        sprite_capacity-= 25*1.5
        
    return render_template('inventory.html',recommend_drink=recommend_drink, coke_capacity=coke_capacity, sprite_capacity=sprite_capacity )
# return make_response('')
#while True:
    #   sleep(1)

   # except:
    #    import traceback
    #    traceback.print_exc()
   # finally:
    #    for p in range(len(processes)):
     #       processes[p].terminate()

      #  print("\n\nFinished\n\n")

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=5000, debug=True)
   
