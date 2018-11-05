import requests
import datetime
import os
import schedule
import time


def download_file(cctv, min_back,unix_timestamp):
   ###Create directory if !exist
    print(datetime.datetime.now())
    directory= "new_videos"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for vid in cctv:
        
        ###get unix time
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        sec=min_back * 60 # e.g. 5 min * 60 seconds
        unix_time = int(unix_timestamp - (sec))
        u_time= str(unix_time) + "-" + str(sec) + ".mp4"
        
        url= vid + "/archive-" + u_time 

        ###download video
        local_filename ="new_videos/" + vid.split(".co.id/")[1] + "-" + u_time 
        
        r = requests.get(url, stream=True)
        
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024): 
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    
    return datetime.datetime.now()

def job():
    ###func takes in list and duration of video in minutes
    current_time = datetime.datetime.now(datetime.timezone.utc)
    unix_timestamp = current_time.timestamp()
    print(download_file(cctv,5,unix_timestamp)) 
    
	
    
####list of all cameras
cctv= ["http://cctv.balitower.co.id/Utan-Kayu-Selatan-004-704477_2", "http://cctv.balitower.co.id/Pluit-035-701162_3","http://cctv.balitower.co.id/Cipinang-Cempedak-006-704139_2","http://cctv.balitower.co.id/Kebon-Jeruk-004-702037_4","http://cctv.balitower.co.id/Kebon-Jeruk-004-702037_3","http://cctv.balitower.co.id/Kebon-Jeruk-004-702037_2","http://cctv.balitower.co.id/Kebon-Jeruk-004-702037_1","http://cctv.balitower.co.id/Pluit-035-701162_3","http://cctv.balitower.co.id/Gambir-014-700307_7","http://cctv.balitower.co.id/Medan-Merdeka-Selatan-C01-506821_1","http://cctv.balitower.co.id/Menteng-001-700123_5","http://cctv.balitower.co.id/Utan-Kayu-Utara-002-704034_4"]

schedule.every(5).minutes.do(job)

while 1:
    schedule.run_pending()
    time.sleep(1)
