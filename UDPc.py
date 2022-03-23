
def main():
    import cv2, imutils, socket
    import numpy as np
    import base64
    import time

    import argparse
    parser = argparse.ArgumentParser('client')
    parser.add_argument('-i', type = str, default = '127.0.0.1', help='frame')
    parser.add_argument('-p', type = int, default = 1, help='frame')
    parser.add_argument('-f', type = int, default = 1, help='frame')
    parser.add_argument('-s', type = int, default = 1, help='frame')
    flags = parser.parse_args()

    import mediapipe as mp
    # Getting mediapipe: Hands ready
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    BUFF_SIZE = 65536
    UDP_IP = flags.i
    UDP_PORT = flags.p
    socket_address = (UDP_IP, UDP_PORT)
    print(f"UDP target IP: {UDP_IP}")
    print(f"UDP target port: {UDP_PORT}")

    if flags.s == 0:
        SOURCE = 0
    else:
        SOURCE = 'v'+str(flags.f)+'.mp4'

    ClientSocket = socket.socket(socket.AF_INET,     # Internet
                          socket.SOCK_DGRAM)  # UDP
    #  | socket.SOCK_NONBLOCK
    ClientSocket.setblocking(0)

    vid = cv2.VideoCapture(SOURCE)
    W,H = [(320,180),(320,240),(640,480),(480,360)][2]
    dim = (W,H)
    screen_coeff=1.5

    def movements(frame,left_wait,right_wait,rotate_wait,down_wait):
        # left_wait = right_wait = rotate_wait = down_wait = 0
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    if id == 0:
                        x = []
                        y = []
                    x.append(int((lm.x) * w))
                    y.append(int((1 - lm.y) * h))

                    # This will track the hand gestures
                    if len(y) > 20:
                        if (x[0] > x[3] > x[4]) and not(y[20] > y[17]):
                            left_wait += 1
                        if not(x[0] > x[3] > x[4]) and (y[20] > y[17]):
                            right_wait += 1
                        if (x[0] > x[3] > x[4]) and (y[20] > y[17]):
                            rotate_wait += 1
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        else:
            down_wait += 1
        return frame,left_wait,right_wait,rotate_wait,down_wait

    def controlling(out,left_wait,right_wait,rotate_wait,down_wait):
        fall_speed = 0
        fall_speed_real = 0
        if 1:
            # "if you gesture to the LEFT for at least 4 frames, piece move LEFT"
            if left_wait >= 2:
                # if there is gestyre then decrease fall speed
                out = b'left'
                # fall_speed = fall_speed_real
                # current_piece.x -= 1
                # if not (valid_space(current_piece, grid)):
                #     current_piece.x += 1
                left_wait = right_wait = rotate_wait = down_wait = 0

            # "if you gesture to the RIGHT for at least 4 frames, piece move RIGHT"
            if right_wait >= 2:
                out = b'righ'
                # fall_speed = fall_speed_real
                # current_piece.x += 1
                # if not (valid_space(current_piece, grid)):
                #     current_piece.x -= 1
                left_wait = right_wait = rotate_wait = down_wait = 0

            # "if you gesture to ROTATE  for at least 4 frames, piece ROTATES"
            if rotate_wait >= 3:
                out = b'rota'
                # fall_speed = fall_speed_real
                # current_piece.rotation += 1
                # if not (valid_space(current_piece, grid)):
                #     current_piece.rotation -= 1
                left_wait = right_wait = rotate_wait = down_wait = 0

            # "if you gesture to go DOWN (no hand on the screen) for at least 5 frames, piece go DOWN (moves very fast)"
            if down_wait >= 3:
                out = b'down'
                # if there is no gestyre then increas fall speed
                # fall_speed = fall_speed_down
                left_wait = right_wait = rotate_wait = down_wait = 0
        return out,left_wait,right_wait,rotate_wait,down_wait

    left_wait = right_wait = rotate_wait = down_wait = 0
    out = b'None'
    zero_frame = np.zeros([H,W,3])
    success = False

    fps,st,frames_to_count,cnt = (0,0,20,0)
    STATUS = b'inac'
    while True:
        # READ A FRAME
        success,frame_out = vid.read() # read from camera
        if success:
            frame_out = cv2.flip(frame_out,1) # flip horizontally
        else:
            frame_out = zero_frame # return zero frame

        # RESIZE A FRAME
        frame_out = cv2.resize(frame_out, dim, interpolation = cv2.INTER_AREA)
        frame_out = imutils.resize(frame_out,width=320)

        # GESTURE RECKOGNITION
        if STATUS == b'acti':
            frame_out,left_wait,right_wait,rotate_wait,down_wait = movements(frame_out,left_wait,right_wait,rotate_wait,down_wait)
            out,left_wait,right_wait,rotate_wait,down_wait = controlling(out,left_wait,right_wait,rotate_wait,down_wait)
    #         frame_out = cv2.putText(frame_out,str(out),(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            STATUS = b'inac'

        # ENCODE AND SEND A FRAME
        encoded,buffer = cv2.imencode('.jpg',frame_out,[cv2.IMWRITE_JPEG_QUALITY,25])
        message = base64.b64encode(buffer)
        ClientSocket.sendto(message+out,socket_address)

        # RECIEVE AND RESIZE A FRAME
        try:
            packet, sender_addr = ClientSocket.recvfrom(BUFF_SIZE) # recieve
            STATUS = packet[-4:]
            data = base64.b64decode(packet,' /') # decode
            npdata = np.fromstring(data,dtype=np.uint8) # decode
            frame_in = cv2.imdecode(npdata,1) # decode
            frame_in = cv2.putText(frame_in,'FPS_1: '+str(fps),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2) # write fps for client
            # resize output image
            shape = frame_in.shape
            dim_show = (int(shape[1]*screen_coeff),int(shape[0]*screen_coeff))
            frame_in = cv2.resize(frame_in, dim_show, interpolation = cv2.INTER_AREA)
            # show on the screen
            cv2.imshow("CLIENT SIDE PROCESSED VIDEO",frame_in)
        except:
            None

        # for stable FPS
        key = cv2.waitKey(1) & 0xFF
        # to count FPS
        if cnt == frames_to_count:
            try:
                fps = round(frames_to_count/(time.time()-st))
                st=time.time()
                cnt=0
            except:
                pass    
        cnt+=1

if __name__ == "__main__":
    main()
