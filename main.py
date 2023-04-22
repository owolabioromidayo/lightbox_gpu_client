import sys, time, traceback
import socketio
from model_broadcast import ModelBroadcast

Q_SERVER_URL = "http://localhost:3003"
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2ODMyNDI2OTIsImlhdCI6MTY4MDY1MDY5MiwiaWQiOjd9.nDqOfB9j-id0-3LMcF6YXCipya_ZBfALbM5Dc39tXfc"

BUSY  = False
CONNECTED = False
ABORT = False
AUTHENTICATED = False

broadcast = ModelBroadcast(api_token = API_TOKEN , test=False)
sio = socketio.Client(logger=True, engineio_logger=True)

@sio.event
def connect():
    print('Connection established with queue server.')
    CONNECTED = True 


@sio.on("broadcast information")
def on_broadcast_formation(data):
    global ABORT
    if data != "Authenticated":
        print(data)
        print("Invalid details were shared. Aborting...")
        ABORT = True
        sio.disconnect()
        return 
    else:
        global AUTHENTICATED
        AUTHENTICATED = True
    
    print('I recv ', data)


@sio.on("no task")
def no_task():
    print('No task')

@sio.on(f"execute task")
def exec_task(_json):

    global BUSY
    BUSY = True
    # print('Executing', _json, "on ", api_id)

    model, args, timestamp = _json["model"], _json["args"], _json["timestamp"]

    if model in broadcast.broadcast["models"].keys():
        try:
            _response = broadcast.broadcast["models"][model]["func"](args)
            sio.emit("done task", {"_response": _response, "timestamp": timestamp } )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("An error occured")
            sio.emit("aborted task", {"error": str(e), "timestamp" : timestamp} )

    else:
        sio.emit("aborted task", {"error": "Model does not exist.", "timestamp" : timestamp} )

    BUSY = False
    #all relevant procs here


@sio.event
def disconnect():
    print('disconnected from server')
    CONNECTED = False 


def get_work_background_task():
    while True:
        try:
            print("Busy: ", BUSY)
            if not BUSY:
                print("Sending out that i dont have work")
                sio.emit('get work', {"_id": API_TOKEN})
            for i in range(20) :  sio.sleep(1) 
        except Exception as e:
            print(e)
            sys.exit()



def main():
    global ABORT
    sio.connect(Q_SERVER_URL, wait_timeout=10)
    # sio.wait()
    sio.emit('broadcast information', {"data": broadcast.get_desc()})
    # sio.wait()
    time.sleep(10)
    if ABORT  == True:
        print("Aborting...")
        sys.exit()
    if AUTHENTICATED == True:
        sio.start_background_task(get_work_background_task)
        # sio.wait()
        try:
            while True:
                time.sleep(1)
        except Exception as e:
            print(e)
            sys.exit()
  

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sio.disconnect()
        sys.exit()










    
