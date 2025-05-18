import pygame
import os
import random                          #랜덤 필요
import logging
import sys


pygame.init()
pygame.font.init()
pygame.mixer.init()

def init_game():
    global yellow, skill, REDS, SPAWN, BOSS, platforms,WIDTH, HEIGHT,MAIN_CHAR_HEIGHT,MAIN_CHAR_WIDTH
    global SKILLDMG, MYDMG, yellow_health, Maxhealth, player_level, player_xp, xp_to_next_level
    global current_map_index, current_map, chgbg, monch,MONSTER_WIDTH, MONSTER_HEIGHT
    global itemx, itemy, dropswitch, dropitem, ITEM_LEFTIMAGE, ITEMrect, ITEMS, Item_Weights
    global LRSWITCH, critical, ihurt, Money, monmv_time, last_jump_time
    global yellow_is_jumping, yellow_y_velocity, on_ground, yellow_feet,monster_speed
    global GRAVITY, JUMP_POWER, GROUND_Y, BOSSPO,STARTX, STARTY, SKWIDTH,VEL, MON_VEL, BULLET_VEL
    global monster_healths, MAX_monsterHP, monster_directions,map_names,on_platform
    global yellow_bullets, damage_numbers, mydamage_numbers, skill_effects
    global bullet, monswitch, chgbg, alldeadsw, deadcount,monster_platforms,WIN
    global quest_frame_visible, frame_blink_counter,LEVEL_UP_DISPLAY_TIME, LEVEL_UP_DURATION
    global quest_font, active_quests, all_quests,skillget,monrd,ITEM_HEIGHT, ITEM_WIDTH
    global R_portal, rect_x, rect_y, rect_size,BLACK, WHITE, RED, GREEN, YELLOW
    global switch, MONMV_COOLDOWN, JUMP_COOLDOWN, rdset, money, droprd,hit
 
    
    # 게임 기본 설정
    WIDTH, HEIGHT = 900, 600               #전체 화면의 크기
    STARTX, STARTY = 100, 100              #주인공의 시작위치pi
    SKWIDTH = 100                          #스킬의 사정거리
    VEL = 5                                #이동 속도
    MON_VEL = 1                            #몬스터 이동속도
    BULLET_VEL = 10                        #총알 속도
    SKILLDMG = [1000,2000,3000,4000,5000]  #스킬별로 데미지 분리
    MYDMG = 10                              #내가 받는 데미지 
    # MAX_BULLETS = 3

    MAIN_CHAR_WIDTH, MAIN_CHAR_HEIGHT = 100, 100   #캐릭터의 크기
    MONSTER_WIDTH, MONSTER_HEIGHT = 100, 100       #몬스터의 크기
    ITEM_WIDTH, ITEM_HEIGHT = 70,70                #아이템의 크기

    # 색상 지정 변수
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)

    # 게임 상태 변수
    switch = 1                       #스킬 변경 스위치
    chgbg = 0                        #배경 변경 스위치
    MONMV_COOLDOWN = 1000            #몬스터 움직임 변화 쿨다운
    JUMP_COOLDOWN = 500              #점프와 점프 사이의 간격
    last_jump_time = 0               #점프뛴지 얼마나 됐는지 저장
    monmv_time = 0                   #몬스터가 멈춘지 얼마나 됐는지 저장
    rdset = 0                        #랜덤이 일어나는 변수
    money = 0                        #돈을 저장하는 변수
    dropitem = 0                     #떨어지는 아이템 사진 외부 폴더 저장 변수
    droprd = [0,0,0,0,0]             #떨어진 아이템 랜덤 변수값
    dropswitch = [False,False,False,False,False]#떨어졌는지 확인 스위치
    alldeadsw = False                #모두 죽었는지 확인 스위치
    deadcount = 0                    #죽은 횟수 더하는 변수   
    monswitch = [True,True,True,True,True] #각 몬스터가 죽었는지 확인하는 스위치
    itemx = [0,0,0,0,0]              #떨어지는 위치 저장#
    itemy = [0,0,0,0,0]              
    critical = 1                     #치명타를 저장하는 변수
    monch = 1                        #몬스터 바꿀때 사용하는 스위치
    skillget = 1                     #스킬 아이템을 얻은 갯수 저장변수
    monrd = [1,1,1,1,1]              #몬스터가 랜덤하게 움직일지 정하는 변수
    ihurt = False                    #적과의 충돌시 on
    Money = 0                        #돈 저장변수
    LRSWITCH = 'r'                   #쳐다보는 방향 저장
    hit = False

    GRAVITY = 1                      #중력 만큼 떨어짐
    JUMP_POWER = 15                  #뛰는 높이
    GROUND_Y = HEIGHT - 20 - MAIN_CHAR_HEIGHT #땅의 위치
    BOSSPO = 8                       #보스위치 8

    #캐릭터 변수
    yellow_is_jumping = False        #캐릭터가 점프중인가? 스윛치
    yellow_y_velocity = 0            #캐릭터의 떨어지는 위치
    Maxhealth = 300                  #캐릭터 최대체력
    yellow_health = Maxhealth        #캐릭터의 체력
    on_ground = False                #땅인가? 스위치
    monster_speed = 1                #몬스터의 속도
    on_platform = [False,False,False,False,False]              #발판인가? 스위치
    player_level = 1                 #캐릭터 레벨 초기변수
    player_xp = 0                    # 레벨업 위한 xp 변수
    xp_to_next_level = 100           # 레벨업을 위해 필요한 xp

    #퀘스트창 변수
    quest_frame_visible = False      #퀘스트 창이 현재 보이는지 확인하는 스위치
    frame_blink_counter = 0          

    map_names = ["Forest", "Hill","Island","Coast City", "Desert Coast","Deep Wood", "Ugdrasil","Dark Forest","Ruins"]  # 원하는 맵 이름 추가

    current_map_index = 0            # 현재 맵 시작은 Forest
    current_map = map_names[current_map_index]    #현재맵의 인덱스 변수 

    REDS = [                         #몬스터 위치 및 크기
    pygame.Rect(700, 500,MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(190, 80, MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(600, 160, MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(200, 260, MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(650, 360, MONSTER_WIDTH, MONSTER_HEIGHT)
    ]

    SPAWN = [                        # 몬스터들의 새로 출연할때 필요한 몬스터 위치
    pygame.Rect(700, 500,MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(190, 80, MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(600, 160, MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(200, 260, MONSTER_WIDTH, MONSTER_HEIGHT),
    pygame.Rect(650, 360, MONSTER_WIDTH, MONSTER_HEIGHT),
    ]

    BOSS = [                         # 보스 몬스터 위치와 크기
    pygame.Rect(900, 90, 300, 500)
    ]

    ITEMrect = [                         # 아이템이랑 충돌할떄 필요한 몬스터 위치
    pygame.Rect(700, 500,ITEM_WIDTH, ITEM_HEIGHT),
    pygame.Rect(190, 80, ITEM_WIDTH, ITEM_HEIGHT),
    pygame.Rect(600, 160, ITEM_WIDTH, ITEM_HEIGHT),
    pygame.Rect(200, 260, ITEM_WIDTH, ITEM_HEIGHT),
    pygame.Rect(650, 360, ITEM_WIDTH, ITEM_HEIGHT)
    ]

    platforms = [                        #발판들의 위치 크기
        pygame.Rect(0, HEIGHT, 900, 20),
        pygame.Rect(80, 170, 300, 20),
        pygame.Rect(500, 250, 300, 20),
        pygame.Rect(100, 350, 300, 20),
        pygame.Rect(500, 450, 300, 20)
    ]
    # 사각형 설정
    rect_size = 70
    rect_y = 10
    rect_x = 540  # 초기 위치

    ITEMS = [0,1,2,3,4]                    #떨어지는 아이템 종류
    Item_Weights = [40,30,10,10,10]        #떨어지는 아이템의 각각의 확률
    ITEM_LEFTIMAGE =[0,0,0,0,0]            #안먹고 남아 있는 아이템 종류 이미지 저장  
    LEVEL_UP_DISPLAY_TIME = 0               #레벨업 디스플레이 초기 변수
    LEVEL_UP_DURATION = 2000                #레벨업 디스플레이 지속시간

    monster_platforms = [0, 1, 2, 3, 4]    #몬스터가 정해질 발판 위치 
    R_portal = pygame.Rect(800, GROUND_Y, 100, 100) #포탈의 위치

    yellow = pygame.Rect(STARTX, GROUND_Y ,  MAIN_CHAR_WIDTH, MAIN_CHAR_HEIGHT)  #주인공의 시작위치와 크기를 저장한 변수
    skill = pygame.Rect(100, 100, 80, 80)                                        #스킬의 시작위치와 크기 (초기 기본값)
    monster_healths = [10000 for _ in REDS]                                      #각 몬스터들의 기본체력을 설정
    MAX_monsterHP = [10000 for _ in REDS]                                        #몬스터 최대 체력 저장 (체력바를 위해 필요)
    monster_directions = [-1 for _ in REDS]                                       #각 몬스터들이 보는 방향을 설정                                                         
    yellow_bullets = []                                                          #캐릭터의 스킬 저장변수
    damage_numbers = []                                                          #데미지 숫자가 들어갈 저장변수
    mydamage_numbers = []                                                        #주인공에 대한 데미지 숫자가 들어갈 저장변수
    skill_effects = []                                                           #적에게 맞은 후 나오는 스킬 효과의 저장변수
    yellow_feet = pygame.Rect(yellow.x, yellow.y + yellow.height, yellow.width,1) # 주인공 발의 위치 = 발판과 충돌의 정확성위해
    
    # 창의 크기와 창에 쓰이는 글씨 
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Maple Story!")


def init_assets():                                                          #외부 파일들을 불러오는 힘수들
    global YELLOW_LEFT, YELLOW_RIGHT, YELLOW_SKILL, SPACE, DYELLOW_LEFT, DYELLOW_RIGHT
    global HEALTH_FONT, WINNER_FONT, PORTAL_IMAGE, DROPITEM_IMAGE, UI_BACKGROUND,UI_BACKGROUNDBASE
    global SKILL_EFFECT_IMAGE, MONSTER, PLATFORM_IMAGE, effect_positions,BASE_FONT,quest_font
    global SELF_EFFECT1, SELF_EFFECT2, SELF_EFFECT3, SELF_EFFECT4, SELF_EFFECT5, logging

    BASE_FONT = pygame.font.Font("Assets/Maplestory Bold.ttf", 20)           #폰트들
    HEALTH_FONT = pygame.font.Font("Assets/Maplestory Bold.ttf", 40)
    WINNER_FONT = pygame.font.Font("Assets/Maplestory Bold.ttf", 80)
    quest_font = pygame.font.Font("Assets/Maplestory Bold.ttf", 28)

    YELLOW_LEFT = pygame.transform.scale(                                    #주인공 왼쪽모습
        pygame.image.load(os.path.join('Assets', 'maincR.png')), (110, 110)).convert_alpha()
    YELLOW_RIGHT = pygame.transform.scale(                                   #주인공 오른쪽모습
        pygame.image.load(os.path.join('Assets', 'maincL.png')), (110, 110)).convert_alpha()
    
    DYELLOW_LEFT = pygame.transform.scale(                                   #주인공 맞았을때 왼쪽모습
        pygame.image.load(os.path.join('Assets', 'DmaincR.png')), (110, 110)).convert_alpha()
    DYELLOW_RIGHT = pygame.transform.scale(                                  #주인공 맞았을때 오른쪽모습
        pygame.image.load(os.path.join('Assets', 'DmaincL.png')), (110, 110)).convert_alpha() 

    SKILL_EFFECT_IMAGE = pygame.image.load(os.path.join('Assets','Effect1.png')) #적이 맞았을때 효과 
    SKILL_EFFECT_IMAGE  = pygame.transform.scale(SKILL_EFFECT_IMAGE, (180,180))

    MONSTER_IMAGE = pygame.image.load(os.path.join('Assets', 'monster.png'))     #몬스터들
    MONSTER = pygame.transform.rotate(pygame.transform.scale(MONSTER_IMAGE, (MONSTER_WIDTH, MONSTER_HEIGHT)), 0)

    YELLOW_SKILL = pygame.transform.scale(                                      #날라가는 스킬 
        pygame.image.load(os.path.join('Assets', 'skill1.png')), (80, 80)).convert_alpha()
                                                                                #발판 
    PLATFORM_IMAGE = pygame.image.load(os.path.join('Assets', 'plate.png')).convert_alpha()

    DROPITEM_IMAGE = pygame.transform.scale(                                    #떨어지는 아이템
        pygame.image.load(os.path.join('Assets', 'item1.png')), (80, 80)).convert_alpha()

    SPACE = pygame.transform.scale(                                             #배경
        pygame.image.load(os.path.join('Assets', 'background0.png')), (WIDTH, HEIGHT))
                                                                                #포탈
    PORTAL_IMAGE = pygame.image.load(os.path.join('Assets', 'potal.png')).convert_alpha()

    SELF_EFFECT1 = pygame.transform.scale(                                      #몸에서 나오는 스킬
        pygame.image.load(os.path.join('Assets', 'SelfEffect1.png')), (200,200)).convert_alpha()
    SELF_EFFECT2 = pygame.transform.scale(
        pygame.image.load(os.path.join('Assets', 'SelfEffect2.png')), (200,200)).convert_alpha()
    SELF_EFFECT3 = pygame.transform.scale(
        pygame.image.load(os.path.join('Assets', 'SelfEffect3.png')), (200,200)).convert_alpha()
    SELF_EFFECT4 = pygame.transform.scale(
        pygame.image.load(os.path.join('Assets', 'SelfEffect4.png')), (200,200)).convert_alpha()
    SELF_EFFECT5 = pygame.transform.scale(
        pygame.image.load(os.path.join('Assets', 'SelfEffect5.png')), (200,200)).convert_alpha()

    UI_BACKGROUND = pygame.transform.scale(                                    #스킬 UI 
        pygame.image.load(os.path.join('Assets', 'UIBACK1.png')), (350,70)).convert_alpha()

    UI_BACKGROUNDBASE = pygame.transform.scale(                                #스킬 UI 배경      
        pygame.image.load(os.path.join('Assets', 'UIBACK.png')), (400,100)).convert_alpha()
    
    
    effect_positions = [                                                      #몸에서 나오는 효과들 위치설정
        (SELF_EFFECT1, (yellow.x - 40, yellow.y - 40)),
        (SELF_EFFECT2, (yellow.x - 40, yellow.y - 40)),
        (SELF_EFFECT3, (yellow.x - 40, yellow.y - 40)),
        (SELF_EFFECT4, (yellow.x + 90, yellow.y - 40)),
        (SELF_EFFECT5, (yellow.x - 40, yellow.y - 100))
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("game_log.txt"),  # 로그를 파일로 저장
            logging.StreamHandler(sys.stdout)     # 로그를 콘솔에도 출력
        ]
    )

class Quest:                                      #퀘스트 클래스
    def __init__(self, name, description, objective_type, objective_target, objective_count, xp_reward):
        self.name = name
        self.description = description
        self.objective_type = objective_type
        self. objective_target = objective_target
        self.objective_count = objective_count
        self.xp_reward = xp_reward
        self.progress = 0
        self.completed = False

    def update_progress(self,event_type, target): #퀘스트 진행
        if self.completed:
            return
        if event_type == self.objective_type and target == self.objective_target:
            self.progress += 1
            if self.progress >= self.objective_count:
                self.completed = True

    def claim_reward(self):
        if self.completed:
            award_xp(self.xp_reward)
            return True
        return False

all_quests = [
        Quest("First Hunt", "Defeat 3 monsters", "kill", "monster", 3, 50),
        Quest("Second Hunt", "Defeat 5 hunters", "kill", "monster", 5, 100 ),
        Quest("Third Quest", "Defeat 10 hunters", "kill", "monster", 10, 150),
        Quest("Forth Quest", "Defeat 15 hunters", "kill", "monster", 15, 200)
    ]
current_quest_index = 0
active_quests = [all_quests[current_quest_index]] if all_quests else []

#LEVELUP MESSAGE DEFINITION
def draw_level_up_message():                                     #레벨업 알려주는 글 출력 함수
    if clock - LEVEL_UP_DISPLAY_TIME <LEVEL_UP_DURATION:
        font = pygame.font.Font("Assets/Maplestory Bold.ttf", 80)
        text = font.render("LEVEL UP!", True, (255,215,0))
        text_rect = text.get_rect(center=(WIDTH//2,HEIGHT//2))
        WIN.blit(text, text_rect)

def draw_quest_frame():                                          # 퀘스트창 그리기 함수
    global frame_blink_counter
    frame_width = 400
    frame_height = 80
    frame_x = 20
    frame_y = 130
    border_color = (255, 215, 0)
    bg_color = (30, 30, 30, 200)

    frame_surface = pygame.Surface((frame_width, frame_height), pygame.SRCALPHA)
    frame_surface.fill(bg_color)

    border_width = 4 if any(q.completed for q in active_quests) else 2
    pygame.draw.rect(frame_surface, border_color, (0, 0, frame_width, frame_height), border_width)

    y_offset = 10
    line_height = 25

    for quest in active_quests:
        if y_offset + line_height > frame_height:
            break

        status = 'Done!!' if quest.completed else f"{quest.progress}/{quest.objective_count}"
        quest_text = quest_font.render(f"Quest: {quest.name}[{status}]", True, YELLOW if quest.completed else WHITE)

        frame_surface.blit(quest_text, (10, y_offset))
        y_offset += line_height

    WIN.blit(frame_surface, (frame_x, frame_y))
    frame_blink_counter += 1

class SkillEffect:                                     # 맞은후 나오는 스킬을 표현하는 클래스
    def __init__(self, x ,y):
        self.x = x
        self.y = y - 80
        self.lifetime = 15                             # 스킬의 지속시간

    def draw(self,win):                                # 스킬 그리기
        win.blit(SKILL_EFFECT_IMAGE, (self.x, self.y - 80))

    def update(self):
        self.lifetime -= 1                             # 지속시간 경과

class DamageNumber:                                    # 데미지를 숫자로 표시하는 클래스 
    def __init__(self, x, y, value, color=(255,255,0)):
        self.x = x
        self.y = y
        self.value = value
        self.color = color
        self.alpha = 255
        self.font = pygame.font.Font("Assets/Maplestory Bold.ttf", 40)
        self.lifetime = 30                             # 데미지 숫자 지속시간

    def draw (self, win):                              # 데미지 그리기
        text = self.font.render(str(self.value), True, self.color)
        text.set_alpha(self.alpha)                      
        win.blit(text, (self.x, self.y))               # 출력

    def update(self):                                  # 사라지는 시간들 업데이트
        self.y -= 2
        self.alpha -= 8
        self.lifetime -= 1
    
def update_effect_postions():
    global effect_positions
    effect_positions = [                                    #효과의 위치를 현재캐릭의 포지션으로 다시 저장
        (SELF_EFFECT1, (yellow.x - 40, yellow.y - 40)),
        (SELF_EFFECT2, (yellow.x - 40, yellow.y - 40)),
        (SELF_EFFECT3, (yellow.x - 40, yellow.y - 40)),
        (SELF_EFFECT4, (yellow.x + 90, yellow.y - 40)),
        (SELF_EFFECT5, (yellow.x - 40, yellow.y - 100))
    ]

def draw_window():                                     #화면에 출력하는 함수
    global bullet
    WIN.blit(SPACE, (0, 0))                            #배경 출력
    
    update_effect_postions()  
    WIN.blit(UI_BACKGROUNDBASE, (515,-5))                   #UI 배경 그리기
    WIN.blit(UI_BACKGROUND, (540,10))                       #UI 그리기 
    
    pygame.draw.rect(WIN, RED, (rect_x, rect_y, rect_size, rect_size), 2)

    for bullet in yellow_bullets:                           #캐릭터 주변 효과 그리기
        WIN.blit(YELLOW_SKILL, (bullet.x, bullet.y-80))      #날라가는 스킬
        max_effects = min(skillget, len(effect_positions))  #몸주변 스킬 총갯수와 내가 얻은 효과중 적은쪽의 것을 선택
        for i in range(max_effects):                        #그 갯수만큼만 스킬이 보임
            effect_image, position = effect_positions[i]    #스킬 레벨이 올라갈수록 많은 효과가 보이는 효과
            if LRSWITCH == 'l' and i == 3:                  #왼쪽을 볼때 몸주변의 4번째 효과는 좌우 반전후 좌표 변경출력
                flipped_image = pygame.transform.flip(SELF_EFFECT4, True, False)
                WIN.blit(flipped_image,(yellow.x - 160, yellow.y - 40))    
            else:                                           #오른쪽 볼때 몸주변의 효과 출력
                WIN.blit(effect_image, position)

    if LRSWITCH == 'l':                                         #왼쪽을 볼때 
        if ihurt :                                              #내가 맞고 있을때
            WIN.blit(DYELLOW_LEFT, (yellow.x, yellow.y))  
        else :                                                  #맞지 않고 있을때
            WIN.blit(YELLOW_LEFT, (yellow.x, yellow.y))         #왼쪽 보는 사진 출력
    else:                                                       #오른쪽을 볼때
        if ihurt:                                               #내가 맞고 있을떄
            WIN.blit(DYELLOW_RIGHT, (yellow.x, yellow.y))
        else :                                                  #맞지 않고 있을때
            WIN.blit(YELLOW_RIGHT, (yellow.x, yellow.y))        #오른쪽을 보는 사진 출력하기

    for plat in platforms:                                      #발판 그리기
        scaled_platform_image = pygame.transform.scale(PLATFORM_IMAGE, (plat.width, plat.height))
        WIN.blit(scaled_platform_image, (plat.x, plat.y))

    for i,monster in enumerate(REDS):                           #총 몬스터 갯수만큼 불러오기
        if monswitch[i] and chgbg != BOSSPO:                    #몬스터가 살아있거나 보스가 아니면
            if monster_directions[i] == 1:                      #몬스터 보는 방향 오른쪽이라면 
                flipped_image = pygame.transform.flip(MONSTER, True, False)         #좌우 반전 
                WIN.blit(flipped_image,(monster.x, monster.y))                      #출력
            else:
                WIN.blit(MONSTER, (monster.x, monster.y))                           #그냥 출력
            pygame.draw.rect(WIN,RED,(monster.x,monster.y-30, 100, 20) )              #몬스터 에너지바 배경
            if monster_healths[i] > 0:                                                #몬스터가 살아있으면 
                health_ratio = monster_healths[i] / MAX_monsterHP[i]                  #몬스터 실제 HP바
                pygame.draw.rect(WIN,GREEN,(monster.x,monster.y-30,100 * health_ratio,20)) 
        elif monswitch[i] and chgbg == BOSSPO:                                       #보스는 크기가 커서 위치를 조금 변경
            WIN.blit(MONSTER, (monster.x, monster.y))                            #보스 출력
            pygame.draw.rect(WIN,RED,(monster.x+50,monster.y, 200, 20) )          #보스 에너지바
            if monster_healths[i] > 0:                                               #보스가 살아있으면
                health_ratio = monster_healths[i] / MAX_monsterHP[i]                   #보스 실제 에너지 계산
                pygame.draw.rect(WIN,GREEN,(monster.x+50,monster.y, 200 * health_ratio,20))

    draw_health_bar()                                                               #몬스터 체력바 그리기

    yellow_health_text = BASE_FONT.render("Health: " + str(yellow_health), 1, WHITE) #캐릭터의 에너지를 화면제일 왼쪽위에 출력
    WIN.blit(yellow_health_text, (10, 10))

    Map_position_text = BASE_FONT.render("MAP: " + map_names[chgbg], 1, WHITE)       #맵의 이름을 화면제일 위에 에너지 밑에 출력
    WIN.blit(Map_position_text, (10, 35))

    Money_text = BASE_FONT.render("MESO: " + str(money), 1, WHITE)                   #메소(돈)을 화면제일 위에 에너지 밑에 출력
    WIN.blit(Money_text, (200, 35))

    for dmg in damage_numbers[:]:                                                   #몬스터 데미지 글씨 출력
        dmg.draw(WIN)

    for mydmg in mydamage_numbers[:]:                                               #내가 맞는 데미지 글씨 출력
        mydmg.draw(WIN)

    for effect in skill_effects[:]:                                                 #스킬 나가는 효과 출력
        effect.draw(WIN)
        
    if chgbg < len(map_names) - 1 and alldeadsw:                                    # 오른쪽으로 이동 가능하고 몹이 없을 경우 포탈 그리기
        scaled_potal_image = pygame.transform.scale(PORTAL_IMAGE, (R_portal.width, R_portal.height))
        WIN.blit(scaled_potal_image, (R_portal.x, R_portal.y))

    for i in range(len(REDS)):                                                      #떨어진 아이템 출력
        if dropswitch[i]:
            WIN.blit(ITEM_LEFTIMAGE[i],(itemx[i], itemy[i]+20))

    level_text = BASE_FONT.render(f"Level: {player_level}", 1, WHITE)               #레벨 과 경험치 화면에 출력
    xp_text = BASE_FONT.render(f"XP: {player_xp}/{xp_to_next_level}", 1, WHITE)
    WIN.blit(level_text, (350, 10))
    WIN.blit(xp_text, (350, 40))

    if player_level >= 2:
        draw_level_up_message()

    if quest_frame_visible:
        draw_quest_frame()

    pygame.display.update()

def draw_health_bar():                                       #몬스터들 체력바 그리기
    pygame.draw.rect(WIN,RED,(10,12, Maxhealth, 20) )
    if yellow_health > 0:
        health_ratio = yellow_health / Maxhealth             #몬스터들 체력바 계산
        pygame.draw.rect(WIN,GREEN,(10,12,Maxhealth * health_ratio,20))

def collid_portal():                                        #포탈 충돌처리
    global current_map, chgbg, monch
    chgbg += 1                                              #배경변수 숫자 증가
    monch += 1                                              #몬스터종류변수 숫자 증가
    Newmonster()                                            #몬스터 새로 스폰 
    Newitem()                                               #아이템 새로 스폰
    yellow.x = STARTX                                       #캐릭 시작위치 저장
    current_map = map_names[chgbg]                          #문자열 갱신


def Newmonster():                                           #몬스터 스폰 함수
    global monster_healths,MAX_monsterHP,alldeadsw,deadcount,REDS,MON_VEL
    if chgbg == BOSSPO:                                     #보스 위치면 보스 설정값 입력
        REDS = [rect.copy() for rect in BOSS]               #몹을 보스 데이타로 입력
        monswitch[0] = True                                 #몹의 살아있는 스위치 ON
        alldeadsw = False                                   #모두 죽음의 스위치 OFF
        deadcount = 0                                       #죽인 횟수 초기화
        # MON_VEL = 0.5                                     #보스는 느리게 움직이기 이동속도 0.5로
        monster_healths[0] = 2000000                        #몬스터 에너지를 20만으로 설정
        MAX_monsterHP[0] = monster_healths[0]               #최대 체력 설정

    else:
        REDS = [rect.copy() for rect in SPAWN]              #새 몬스터 
        for i in range(len(SPAWN)):                         #몬스터를 새로 배치
            monswitch[i] = True                             #몹의 살아있는 스위치 ON
            alldeadsw = False                               #모두 죽음의 스위치 OFF
            deadcount = 0                                   #죽인 횟수 초기화
            monster_healths[i] = 20000*monch                #몬스터 에너지를 다시 설정
            MAX_monsterHP[i] = monster_healths[i]
        
def Newitem():                                              #아이템 초기화
    for i in range(len(ITEMrect)):                          #아이템 스위치를 초기화
        dropswitch[i] = False

def award_xp(amount):                                       #xp 보상 함수
    global player_xp
    player_xp += amount
    if player_xp >= xp_to_next_level:
        level_up()

def level_up():                                             #레벨업 함수
    global player_level, player_xp, xp_to_next_level, yellow_health, SKILLDMG
    global LEVEL_UP_DISPLAY_TIME , Maxhealth

    player_level += 1                                       #레벨은 1씩 증가
    player_xp = 0                                           #경험치 초기화
    xp_to_next_level = int(xp_to_next_level*1.1)            #렙업후 다음 렙까지 경치는 1.1배 더 벌어야됨
    yellow_health += 20                                     #렙업시 내 HP +20
    critical += 0.1

    LEVEL_UP_DISPLAY_TIME = clock                           #렙업 화면표시
    print(f"LEVEL UP!! Now level {player_level}")

def handle_keys():                                          #키 조작 함수
    global LRSWITCH, chgbg, yellow_health, ihurt, alldeadsw            
    global yellow_is_jumping, yellow_y_velocity, on_ground, yellow_feet, last_jump_time, monch

    yellow_feet = pygame.Rect(yellow.x, yellow.y + yellow.height, yellow.width,1) # 주인공 발의 위치 = 발판과 충돌의 정확성위해
    
    for i in range(len(REDS)):                              #몹이 있나?
        if deadcount >= len(REDS):                          #죽은 갯수가 몹수랑 같거나 더 많으면 
            alldeadsw = True                                #모두 죽었다 스위치 온

    keys_pressed = pygame.key.get_pressed()
    if keys_pressed[pygame.K_LEFT]:                         # <-- 좌측화살표키를 눌렀을때
        if  yellow.x < 0:                                   # 캐릭이 맵밖으로 벗어날떄
            yellow.x += VEL                                 #캐릭이 벽을 넘지 못하게 반대로 이동
        else: 
            yellow.x -= VEL                                 #정상일때 왼쪽으로 이동
            LRSWITCH = 'l'                                  #좌측 이동 스위치 저장
        
        if yellow.colliderect(R_portal) and alldeadsw:      #포탈에 충돌하는데 몹이 없을때
            collid_portal()                                     

        for plat in platforms:
            if yellow_feet.colliderect(plat):                #캐릭터의 발이 발판에 충돌
                if  plat.left <= yellow.centerx:             #발판안에 내 좌표가 있을때 
                    on_ground = True                         #땅이라고 알림
                else :
                    on_ground = False                        #발판에서 떨어지면 땅이아니라고 알림

        for i, irect in enumerate(ITEMrect):                 #아이템 충돌 확인
            if yellow.colliderect(irect) and dropswitch[i]:  #아이템 충돌하고 아이템 스위치가 온일떄
                get_item(i)                                  #아이템을 얻는 함수 호출
                dropswitch[i] = False                        #얻었으니 아이템 스위치 끄기

    if keys_pressed[pygame.K_RIGHT]:                         # 우측화살표키-->를 눌렀을때
        if  yellow.x >= WIDTH - yellow.width:                #캐릭이 우측 끝에 가면
            yellow.x -= VEL                                  #반대로 움직임
        else:
            yellow.x += VEL                                  #정상이면 우측으로 움직임
            LRSWITCH = 'r'                                   #우측을 본다고 스위치 온
        
        if yellow.colliderect(R_portal) and alldeadsw:       #포탈에 충돌하는데 몹이 없을때
            collid_portal()

        for plat in platforms:                        
            if yellow_feet.colliderect(plat):               #발판에 충돌할때
                if  yellow.centerx < plat.right:            #나의 중심위치가 발판의 우측끝보다 안에 있을때
                    on_ground = True                        #지면이라고 설정
                else :
                    on_ground = False                       #지면이 아니라고 설정

        for i, irect in enumerate(ITEMrect):                #아이템과 충돌
            if yellow.colliderect(irect) and dropswitch[i]: #떨어진 아이템 스위치 on 이면서 아이템과 충돌할때
                get_item(i)
                dropswitch[i] = False                       #떨어진 아이템 스위치 OFF

    if yellow.colliderect(R_portal) and alldeadsw:          #포탈에 충돌하는데 몹이 없을때
        collid_portal()

    for i,monster in enumerate(REDS):                       #캐릭터와 몬스터의 충돌
        if yellow.colliderect(monster) and monswitch[i]:    #몬스터와 충돌시
            mydamage_numbers.append(DamageNumber(yellow.x+yellow.width//2, yellow.y, MYDMG, RED)) #나의 데미지 출력을 위한 숫자 저장
            ihurt = True                                    #충돌중이라고 설정
            yellow_health -= MYDMG                          #에너지에서 10을 뺌
        else :
            ihurt = False                                   #아니면 충돌중이 아니라고 설정

    if not yellow_is_jumping and keys_pressed[pygame.K_LALT] and on_ground:  #점프중이아니면서 and 바닥일때 and 점프를 눌렀을때
        if clock - last_jump_time >= JUMP_COOLDOWN:         #점프쿨다운 시간이 지났는지 확인
            yellow_is_jumping = True                        #점프중 스위치 ON
            yellow_y_velocity = -JUMP_POWER                 #점프파워 높이 만큼 가속도에 저장
            last_jump_time = clock

        if yellow.colliderect(R_portal) and alldeadsw:       #포탈에 충돌하는데 몹이 없을때
            collid_portal()

    if yellow_is_jumping:                                   #점프를 할때
        yellow.y += yellow_y_velocity                       #내 위치에 가속도 더하기
        yellow_y_velocity += GRAVITY                        #가속도에 중력만큼 더하기
        for plat in platforms:                              #플래폼에 충돌하고 + 떨어지고 있을때 + 플랫폼 안에 있다면
            if yellow_feet.colliderect(plat) and yellow.centerx >= plat.left <= plat.right and yellow_y_velocity > 0:
                on_ground = True                            #지면 스위치 on
                yellow_is_jumping = False                   #점프중 스위치 off
                yellow.y = plat.top - yellow.height         #플랫폼의 위치에서 나의 키를 뺀만큼 위치가 나의 y좌표
                yellow_y_velocity = 0                       #낙하 속도 0
                break                                       #점프중 에서 break
                                
    if yellow.y >= GROUND_Y :#땅에 서있을떄 발판에 충돌하지 않을때
        yellow.y = GROUND_Y                                       #내 위치에 땅의 위치 저장
        yellow_is_jumping = False                                 #점프중 스위치 OFF
        on_ground = True                                          #땅 스위치 ON
        yellow_y_velocity = 0                                     #가속도 0

    if not on_ground:                                       #떨어지고 있을때 
        yellow.y += yellow_y_velocity                       #캐릭이 아래로 떨어짐
        yellow_y_velocity += GRAVITY                        #중력속도만큼 

def shoot_bullet():
    if LRSWITCH == 'r':                                     #내가 오른쪽을 보고 있을때 총알의 시작 위치 및 충돌 크기
        bullet = pygame.Rect(yellow.x + yellow.width / 2, yellow.y + yellow.height /2 , 50, 50)
    else:                                                   #내가 ㅇ왼쪽을 보고 있을때 총알의 시작 위치 및 충돌 크기
        bullet = pygame.Rect(yellow.x - yellow.width / 2, yellow.y + yellow.height /2 , 50, 50)
    yellow_bullets.append(bullet)

def handle_bullets():
    global hit,itemx,itemy,deadcount
    hit = False                                             #
    for bullet in yellow_bullets[:]:  
        if (LRSWITCH  == 'r' and bullet.x > yellow.x):      #오른쪽으로 보고 총알은 오른쪽으로,
            bullet.x += BULLET_VEL                          #총알위치는 총알속도만큼 더해서 날라감
        elif (LRSWITCH  == 'r' and bullet.x < yellow.x):    # 내가 방향을 바꿨더라도 쏜총알은 바뀌지 않기
            bullet.x -= BULLET_VEL            
        elif (LRSWITCH  == 'l'and bullet.x <= yellow.x):    #왼쪽으로 보고 총알방향이 왼쪽으로 
            bullet.x -= BULLET_VEL
        elif (LRSWITCH  == 'l'and bullet.x > yellow.x):     # 내가 방향을 바꿨더라도 쏜총알은 바뀌지 않기
            bullet.x += BULLET_VEL

        for i , monster in enumerate(REDS):                 #몬스터 총알과 충돌 확인
            if monster.colliderect(bullet) and monswitch[i]:        #몬스터가 총알과 충돌 그리고 몬스터의 스위치가 ON일때
                if bullet in yellow_bullets:  # 리스트에 bullet이 있는지 확인
                    yellow_bullets.remove(bullet)
         
                monster_healths[i] -= SKILLDMG[switch-1] * critical #데미지 계산 : 스킬 데미지 * 추가 값
                damage_numbers.append(DamageNumber(monster.x+monster.width//2, monster.y,int(SKILLDMG[switch-1] * critical), YELLOW))
                skill_effects.append(SkillEffect(bullet.x, bullet.y + 50)) #몹이 맞았을때 나오는 스킬 위치 저장
                hit = True                                          #맞았다의 스위치 ON
                if monster_healths[i] <= 0:                         #몬스터가 죽었을때 
                    award_xp(30)                                    #경험치 30
                    itemx[i] = monster.x                            #몬스터 위치에 아이템 떨어지게 좌표 입력
                    itemy[i] = monster.y
                    dropswitch[i] = True                            #아이템 떨어지는 스위치 ON
                    ITEMrect[i].x = itemx[i]                        #아이템의 충돌 좌표에 아이템 좌표를 입력
                    ITEMrect[i].y = itemy[i]
                    deadcount += 1                                  #몬스터 죽음 카운트 +1
                    monswitch[i] = False                            #몬스터 스위치 죽었으니 OFF
                    change_dropitem()                               #아이템 떨어지는 사진 랜덤 선택 함수 호출
                    ITEM_LEFTIMAGE[i] = DROPITEM_IMAGE              #떨어진 아이템을 안먹었을 경우를 생각해서 저장
                    droprd[i] = dropitem                            #랜덤으로 떨어진 아이템의 번호 저장 
                    for quest in active_quests:                     #퀘스트 진행
                        quest.update_progress("kill",'monster')

        if not hit and (bullet.x > WIDTH or bullet.x<0):            #총알이 몹에 맞지 않았을때, 화면밖을 벗어나면 
            yellow_bullets.remove(bullet)                           #총알 삭제

def draw_winner(text):                                              #승리 문구 출력
    draw_text = WINNER_FONT.render(text, 1, RED)                    
    WIN.blit(draw_text, (WIDTH/2 - draw_text.get_width() /          #승리 문구 화면 정중앙에 출력
                         2, HEIGHT/2 - draw_text.get_height()/2))
    pygame.display.update()
    pygame.time.delay(5000)                                         # 5초 딜레이

def change_skill_image():                                           #스킬 변환 외부파일 가져오기
    global YELLOW_SKILL
    Skill_image_path = f'skill{switch}.png'                         
    Skill_full_path = os.path.join('Assets', Skill_image_path)
    YELLOW_SKILL = pygame.transform.scale(pygame.image.load(Skill_full_path), (180,180))

def damage_effect():                                                #몹이 맞았을때 나오는 효과 파일 가져오기
    global SKILL_EFFECT_IMAGE 
    Skill_image_path = f'effect{switch}.png'
    Skill_full_path = os.path.join('Assets', Skill_image_path)
    SKILL_EFFECT_IMAGE  = pygame.transform.scale(pygame.image.load(Skill_full_path), (180, 180))

def Monster_movement():                                             #몬스터가 움직이는 함수
    global monster, monmv_time, monster_speed

    for i, monster in enumerate(REDS):
        platform_index = monster_platforms[i]                       #플래폼[]중 하나를 변수에 저장
        platform = platforms[platform_index]            

        rdset = random.randint(1,100)                               #랜덤 1~100까지 중 숫자 하나 선택

        if clock - monmv_time >= MONMV_COOLDOWN and rdset > 70:     #만약 쿨다운이 지나고 랜덤으로 70이상 나올때
            monmv_time = clock                                             
            monrd[i] = random.uniform(0.5,1.5)                      #랜덤으로 0.5 부터 1.5까지 중 값을 선택
            if monrd[i] >= 1.0:                                     #그 값이 1을 넘으면 몬스터의 방향 전환
                monster_directions[i] *= -1

        monster.x += monster_directions[i]* MON_VEL * monch * monrd[i]  #몬스터의 방향에 (몬스터의 속도 X 몬스터의 페이지수 x 몬스터랜덤값)
        monster_speed =  MON_VEL * monch * monrd[i]
        
        if monster.x <= platform.x:                                         #몬스터가 발판안에 있으면
            monster.x = platform.x                                          #그대로 발판위치를 몬스터의 위치로 넣음
            monster_directions[i] = 1                                       #정방향 
        elif monster.x + monster.width >= platform.x + platform.width:      #몬스터가 발판 밖으로 벗어나려하면 
            monster.x = platform.x + platform.width - monster.width         #몬스터의 위치 = 발판의 위치 + 발판의 넓이 - 몬스터 크기 뺴기
            monster_directions[i] = -1                                      #방향 전환

def change_background():                                                    #배경전환
    global SPACE,MONSTER
    bg_path = os.path.join('Assets', f'background{chgbg}.png')              #외부 배경화일의 chgbg번째 사진불러오기
    if os.path.exists(bg_path):                                             #배경화일이 있으면 
        SPACE = pygame.transform.scale(pygame.image.load(bg_path), (WIDTH, HEIGHT))#배경변수에 배경화일을 넣음 

    bg_path = os.path.join('Assets', f'monster{monch}.png')                  #몬스터 파일 다시 불러오기
    if chgbg == BOSSPO:                                                      #보스일경우 이미지 크기를 크게(500,500) 불러오기
        if os.path.exists(bg_path):
            MONSTER = pygame.transform.scale(pygame.image.load(bg_path), (500, 500))
    else:
        if os.path.exists(bg_path):                                          #일반 몹일경우 정해진 몬스터 크기로 불러오기
            MONSTER = pygame.transform.scale(pygame.image.load(bg_path), (MONSTER_WIDTH, MONSTER_HEIGHT))
    
def change_dropitem():                                                       #드랍템 함수
    global DROPITEM_IMAGE,dropitem
    dropitem = random.choices(ITEMS, weights=Item_Weights,k=1)[0]            #선택 랜덤으로 각weight확률로 ITEMS중 하나를 고름
    logging.debug(f'Random choice ITEM:{ITEMS} probability:{Item_Weights} selected:{dropitem}')   
    bg_path = os.path.join('Assets', f'item{dropitem}.png')
    if os.path.exists(bg_path):
        DROPITEM_IMAGE = pygame.transform.scale(pygame.image.load(bg_path), (ITEM_WIDTH, ITEM_HEIGHT))

def change_UI():
    global UI_BACKGROUND,skillget                                            #스킬 먹었을때 UI변경 파일 가져오기
    if skillget > 5:
        skillget = 5                                                         #UI파일이 5개뿐이라 그전까지 불러오기
    bg_path = os.path.join('Assets', f'UIBACK{skillget}.png')                #얻은 스킬갯수 만큼의 번호 불러오기
    if os.path.exists(bg_path):
        UI_BACKGROUND = pygame.transform.scale(pygame.image.load(bg_path), (350, 70))

def get_item(i): 
    global critical, MYDMG, yellow_health,skillget,money                   #줍는 아이템별 효과 결정
    if droprd[i] == 0:                                                     #돈은 메소 + 100
        money += 100
    elif droprd[i] == 1:                                                   #물약은 나의 HP를 최대로 
        yellow_health = Maxhealth
    elif droprd[i] == 2:                                                   #칼을 먹으면 공격력 * 2
        critical += 2
    elif droprd[i] == 3:                                                   #방패는 내가 맞는 데미지 -1 <0이 아닐동안
        if MYDMG != 0:
            MYDMG -= 1
    elif droprd[i] == 4:                                                   #스킬 아이콘은 얻은스킬 + 1
        skillget += 1                                                       
        change_UI()                                                        #UI변경

def seffect_timer():                                                        # 효과 시간 지나면 지우기
    # global effect, dmg, mydmg    
    for dmg in damage_numbers[:]:
        dmg.update()
        if dmg.lifetime <= 0 or dmg.alpha <= 0:
            damage_numbers.remove(dmg)

    for mydmg in mydamage_numbers[:]:
        mydmg.update()
        if mydmg.lifetime <= 0 or mydmg.alpha <= 0:
            mydamage_numbers.remove(mydmg)

    for effect in skill_effects[:]:
        effect.update()
        if effect.lifetime <= 0:
            skill_effects.remove(effect)

def handle_events():                                                        # 공격(Ctrl)과 스킬변경 (1~5) 퀘스트(q)이벤트 함수
    global switch,quest_frame_visible, current_quest_index, active_quests, rect_x
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LCTRL:                                 #공격시 스킬 파일 함수 발사 함수 호출
                change_skill_image()
                shoot_bullet()
            if event.key == pygame.K_1:
                switch = 1
                rect_x = 540
            elif event.key == pygame.K_2 and skillget >= 2:
                switch = 2
                rect_x = 610
            elif event.key == pygame.K_3 and skillget >= 3:
                switch = 3
                rect_x = 680
            elif event.key == pygame.K_4 and skillget >= 4:
                switch = 4
                rect_x = 750
            elif event.key == pygame.K_5 and skillget >= 5:
                switch = 5
                rect_x = 820
            if event.key == pygame.K_q:                                     #퀘스트 키 입력시
                quest_frame_visible = not quest_frame_visible               #퀘스트창 스위치를 반대로 변경 False -> True, 
                if quest_frame_visible:
                    for quest in active_quests[:]:                          #퀘스트 읽어오기
                        if quest.completed:                                 #퀘스트 완료되면
                            if quest.claim_reward():                        #화면에 보상 함수 호출
                                print(f"Quest '{quest.name}' claimed!!")    
                                active_quests.remove(quest)                 #퀘스트 하나 삭제
                                current_quest_index += 1                    #퀘스트 인덱스 +1
                                if current_quest_index < len(all_quests):   #인덱스가 모든 퀘스트 보다 작으면 
                                    active_quests.append(all_quests[current_quest_index])  #활성 퀘스트에 인덱스 퀘스트 저장
                                else:
                                    print("All quests completed")           #크면 모든 퀘스트 완료

def main():
    global clock
    init_game()
    init_assets()
    run = True
    while run:
        clock = pygame.time.get_ticks()                            #게임이 시작한 시간
        handle_events()
        handle_keys()
        damage_effect()
        handle_bullets()
        change_background()
        Monster_movement()
        seffect_timer()
        draw_window()

        if alldeadsw and chgbg == BOSSPO :                  #모든 몹이 죽고 페이지가 보스 페이지라면 
            draw_winner("YOU WINNER!")                      #출력
            break                                           #게임에서 out

        if yellow_health <= 0:                              #내 피가 0보다 작으면
            draw_winner("YOU DIE!")                         #유 다 희
            break                                           #게임에서 out

if __name__ == "__main__":
    main()

