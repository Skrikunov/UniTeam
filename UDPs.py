
##############################################################################
import pygame
import cv2
import random
import math
import numpy as np
# our files
import functions
import shapes


N = 4 # the number of players
FPS = 30 # frames per second
FALL_SPEED = 1.0 # basic fall speed
# video source selection
SOURCE = 'camera'
# SOURCE = 'video'
# cam_idx = 0
cam_idx = -1

layout = functions.get_layout(N) # select screen layout
pygame.font.init() # initialize font

# nearest integer
if N > 1:
    q = 2*math.ceil(N/2)
else:
    q = 1
cell = np.arange(q).reshape(layout[0],layout[1])

# GLOBALS VARS
if SOURCE == 'video':
    file_path = "v1.mp4"
    vid = cv2.VideoCapture(file_path)
    H,W = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
elif SOURCE == 'camera':
    W,H = 320,240
    # W,H = 160,120
s_width = W*layout[0]
s_height = H*layout[1]
block_size = 20

N_ver_blocks = s_height//block_size 
N_hor_blocks = s_width//block_size

play_width = N_hor_blocks*block_size  # meaning 300 // 10 = 30 width per block
play_height = N_ver_blocks*block_size  # meaning 600 // 20 = 30 height per block

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

shapes,shape_colors = shapes.get_shapes_and_colors()

class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0


def create_grid(locked_pos={}):
    grid = [[(0,0,0) for _ in range(N_hor_blocks)] for _ in range(N_ver_blocks)]
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_pos:
                c = locked_pos[(j,i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)
    return positions


def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(N_hor_blocks) if grid[i][j] == (0,0,0)] for i in range(N_ver_blocks)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def get_shape():
    return Piece(random.randint(3,N_hor_blocks-3), 0, random.choice(shapes))


def draw_text_middle(surface, text, size, color):
    font = pygame.font.SysFont("comicsans", size, bold=True)
    label = font.render(text, 1, color)
    surface.blit(label, (top_left_x + play_width /2 - (label.get_width()/2), top_left_y + play_height/2 - label.get_height()/2))


def draw_grid(surface, grid):
    sx = top_left_x
    sy = top_left_y
    for i in range(len(grid)):
        pygame.draw.line(surface, (128,128,128), (sx, sy + i*block_size), (sx+play_width, sy+ i*block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128, 128, 128), (sx + j*block_size, sy),(sx + j*block_size, sy + play_height))


def clear_rows(grid, locked):
    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0,0,0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j,i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)

    return inc

# win = pygame.display.set_mode((s_width, s_height))
##############################################################################

import cv2, imutils, socket
import numpy as np
import base64
import time

import os
from _thread import *

import argparse
parser = argparse.ArgumentParser('server')
parser.add_argument('-i', type = str, default = '127.0.0.1', help='frame')
parser.add_argument('-p', type = int, default = 1, help='frame')
flags = parser.parse_args()

BUFF_SIZE = 65536
UDP_IP = flags.i
UDP_PORT = flags.p
socket_address = (UDP_IP, UDP_PORT)
print(f"UDP target IP: {UDP_IP}")
print(f"UDP target port: {UDP_PORT}")

def main(win):
    ServerSocket = socket.socket(socket.AF_INET,     # Internet
                          socket.SOCK_DGRAM)  # UDP
    ServerSocket.bind(socket_address)

    ADDRESSES = []
    vert_imgs,hor_imgs = 2,2
    W,H = [(320,180),(320,240),(640,480),(480,360)][1]
    dim = (W,H)
    frame_out = np.zeros([H*vert_imgs,W*hor_imgs,3],dtype='uint8')

    ##############################################################################


    RUN = True
    locked_positions = {}
    grid = create_grid(locked_positions)
    clock = pygame.time.Clock() # create a timer
    vert_imgs,hor_imgs = layout[0],layout[1]

    change_piece = False
    current_piece = get_shape()
    next_piece = get_shape()

    fall_speed_real = FALL_SPEED
    fall_speed = fall_speed_real
    fall_speed_down = 0.3

    fall_time = 0
    level_time = 0
    score = 0
    frame_counter=0
    loc_h,loc_w = 0,0
    loc_h1,loc_w1,loc_h2,loc_w2 = 0,0,0,0
    left_wait = right_wait = rotate_wait = down_wait = 0
    ##############################################################################


    fps,st,frames_to_count,cnt = (0,0,20,0)
    ALLOW_CONTROL = False
    STATUS = b'inac'
    while RUN:
        # frame_out = np.zeros([H*vert_imgs,W*hor_imgs,3],dtype='uint8')
        ##############################################################################
        if 1:
            loc = functions.get_image_part(current_piece.x,current_piece.y,N_hor_blocks, N_ver_blocks, n_players_vert = vert_imgs, n_players_hor = hor_imgs)
            if len(loc) == 1:
                loc_h,loc_w = loc[0][0],loc[0][1]
            # elif len(loc) == 2:
            #     loc_h1,loc_w1 = loc[0][0],loc[0][1]
            #     loc_h2,loc_w2 = loc[1][0],loc[1][1]
            if loc_h < vert_imgs and loc_w < hor_imgs:
                player = cell[loc_h,loc_w]
            player_can_control = [player]
        ##############################################################################

        # RECEIVE MESSAGE FROM CLIENT
        packet, sender_addr = ServerSocket.recvfrom(BUFF_SIZE)
        if sender_addr not in ADDRESSES:
            ADDRESSES.append(sender_addr)
        if len(ADDRESSES) > 4:
            ADDRESSES = ADDRESSES[:4]

        ##############################################################################
        for user_number,addr in enumerate(ADDRESSES):
            if addr == sender_addr:
                break
        if user_number in player_can_control:
            STATUS = b'acti'
            command = packet[-4:]
        else:
            command = b'None'
            STATUS = b'inac'
        # print(command,user_number)
        ##############################################################################



        data = base64.b64decode(packet[:-4],' /')
        npdata = np.fromstring(data,dtype=np.uint8)
        frame_in = cv2.imdecode(npdata,1)
        frame_in = cv2.putText(frame_in,'FPS: '+str(fps)+' N='+str(user_number),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        layoutt = [(0,0),(0,1),(1,0),(1,1)]
        x,y = layoutt[user_number]
        frame_out[x*H:(x+1)*H,y*W:(y+1)*W,:] = frame_in
    #     frame_in = cv2.putText(frame_in,'FPS: '+str(fps)+' N='+str(user_number),(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        # frame_out = np.concatenate((frame_in,frame_in),0)
        # frame_out = np.concatenate((frame_out,frame_out),1)

       ##############################################################################
        if 1:
            # draw squazes over the image
            # create a block template
            block = np.ones([block_size,block_size,3])
            for h in range(len(grid)):
                for w in range(len(grid[h])):
                    if grid[h][w] != (0,0,0):
                        # fill colours
                        block[:,:,0] = grid[h][w][0]
                        block[:,:,1] = grid[h][w][1]
                        block[:,:,2] = grid[h][w][2]
                        frame_out[h*block_size:(h+1)*block_size,w*block_size:(w+1)*block_size,:] = block

        # clock.tick(FPS)
        #############################################################################
        cv2.imshow("SERVER SIDE VIDEO",frame_out)

        # SEND MESSAGE TO CLIENT
        encoded,buffer = cv2.imencode('.jpg',frame_out,[cv2.IMWRITE_JPEG_QUALITY,25])
        message = base64.b64encode(buffer)
        ServerSocket.sendto(message+STATUS,sender_addr)


        ##############################################################################
        if 1:
            if command == b'left':
                left_wait += 1
            if command == b'righ':
                right_wait += 1
            if command == b'rota':
                rotate_wait += 1
            if command == b'down':
                down_wait += 1
        if 1:
            grid = create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            level_time += clock.get_rawtime()
            clock.tick()

            # every 10 sec, shapes move 0.02 sec faster (peak at 0.25 sec)
            if level_time/1000 > 10:
                level_time = 0
                if fall_speed_real > 0.25:
                    fall_speed_real -= 0.02

            # if enough time (fall_speed) have passsed, piece moves down 1 block
            if fall_time/1000 > fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not(valid_space(current_piece, grid)) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True

            # CONTROLLING VIA GESTURES
            # quit the game if quit button has been pushed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUN = False
                    cv2.destroyAllWindows()
            # "if you gesture to the LEFT for at least 4 frames, piece move LEFT"
            if left_wait >= 8:
                # if there is gestyre then decrease fall speed
                fall_speed = fall_speed_real
                current_piece.x -= 1
                if not (valid_space(current_piece, grid)):
                    current_piece.x += 1
                left_wait = right_wait = rotate_wait = down_wait = 0

            # "if you gesture to the RIGHT for at least 4 frames, piece move RIGHT"
            if right_wait >= 8:
                fall_speed = fall_speed_real
                current_piece.x += 1
                if not (valid_space(current_piece, grid)):
                    current_piece.x -= 1
                left_wait = right_wait = rotate_wait = down_wait = 0

            # "if you gesture to ROTATE  for at least 4 frames, piece ROTATES"
            if rotate_wait >= 8:
                fall_speed = fall_speed_real
                current_piece.rotation += 1
                if not (valid_space(current_piece, grid)):
                    current_piece.rotation -= 1
                left_wait = right_wait = rotate_wait = down_wait = 0

            # "if you gesture to go DOWN (no hand on the screen) for at least 5 frames, piece go DOWN (moves very fast)"
            if down_wait >= 5:
                # if there is no gestyre then increas fall speed
                fall_speed = fall_speed_real
                left_wait = right_wait = rotate_wait = down_wait = 0

            shape_pos = convert_shape_format(current_piece)

            for i in range(len(shape_pos)):
                x, y = shape_pos[i]
                if y > -1:
                    grid[y][x] = current_piece.color

            if change_piece:
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    locked_positions[p] = current_piece.color
                current_piece = next_piece
                next_piece = get_shape()
                change_piece = False
                fall_speed = fall_speed_real
                # down_wait = 0
                score += clear_rows(grid, locked_positions) * 10

            # update display
            pygame.display.update()

            if check_lost(locked_positions):
                win.fill((0,0,0))
                draw_text_middle(win, "YOU're INVALID!", 80, (255,255,255))
                pygame.display.update()
                pygame.time.delay(1500)
                RUN = False

    #         # PRINT INFO
    #         if frame_counter%15 == 0:
    #             print("FPS = ", str(int(clock.get_fps())))
    #             print("Current player = ",player)
    #             print("Score = ", score)
            frame_counter+=1
        ##############################################################################


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
    
def main_menu(win):
    RUN = True
    while RUN:
        win.fill((0,0,0))
        draw_text_middle(win, 'Press Any Key To Play', 60, (255,255,255))
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('CLOSE THE GAME')
                win.fill((0,0,0))
                draw_text_middle(win, 'Press Any Key To Play', 60, (255,255,255))
                RUN = False
            if event.type == pygame.KEYDOWN:
                win.fill((0,0,0))
                draw_text_middle(win, 'You are playing', 60, (255,255,255))
                print('START THE GAME')
                main(win)
    pygame.display.quit()

# create a window
win = pygame.display.set_mode((s_width, s_height))
# run the main menu
main_menu(win)
