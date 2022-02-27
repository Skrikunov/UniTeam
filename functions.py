def get_layout(N):
    if N == 1:
        a,b = 1,1
    elif N == 2:
        a,b = 2,1
    elif N >= 3 and N <=4:
        a,b = 2,2
    elif N >=5 and N <= 6:
        a,b = 2,3
    return a,b

def get_image_part(x,y,N_hor_blocks, N_ver_blocks, n_players_vert, n_players_hor):
    SAFE_INTERVAL = 1 
    hor_idsx = []

    pixels_per_player_vert = N_ver_blocks/n_players_vert
    pixels_per_player_hor =  N_hor_blocks/n_players_hor

    vert_idx =  int(y//pixels_per_player_vert)
    hor_idx = int(x//pixels_per_player_hor)

    hor_idsx.append(hor_idx)

    if (x+SAFE_INTERVAL)//pixels_per_player_hor != hor_idx:
        if hor_idx < n_players_hor-1:
            hor_idsx.append(hor_idx+1)

    if (x-SAFE_INTERVAL)//pixels_per_player_hor != hor_idx:
        if hor_idx >0 :
            hor_idsx.append(hor_idx-1)
    
    ans = []
    for i, hor in enumerate(hor_idsx):
        ans.append((vert_idx, hor))
    return ans