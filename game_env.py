import numpy as np
import gym
import pygame
import pygame_refactored_AI as game
import math
# from minimap import extract_state_from_screen
# from minimap_debug import draw_minimap_from_state
import time


# 상태 변수 초기값 (게임 전체에서 공유)

class MyGameEnv(gym.Env):
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.step_count = 0
        self.px = 0
        self.py = 0
        self.prev_health = 0
        self.lapclock = pygame.time.Clock()
        self.MAXCOUNTDOWN = 10000  # 또는 원하는 초기값
        self.platform_touched = [0,0,0,0,0]
        # self.minimap_surface = pygame.Surface((40, 40))  # 확대 미니맵 Surface
        self.action_last_time = [0.0] * 10  # 각 액션별 마지막 실행 시각
                # 각 액션에 대한 딜레이 (초 단위)
        self.action_delay_map = {
            0: 0.0,   # 왼쪽 이동
            1: 0.0,   # 오른쪽 이동
            2: 0.5,    # 점프
            3: 0.3,    # 공격
            4: 0.0,    # 대기 (즉시 허용)
            5: 3.0,    # 스킬1
            6: 3.0,    # 스킬2
            7: 3.0,    # 스킬3
            8: 3.0,    # 스킬4
            9: 3.0     # 스킬5
        }
        game.init_game()
        game.init_assets()

    def reset(self):
        game.init_game()
        game.init_assets()
        self.prev_dropswitch = game.dropswitch.copy()
        self.half_health_rewarded = True
        self.half_health_penalized = False  # 다시 감점 받을 수 있도록 초기화
        self.platform_touched = [0,0,0,0,0]

        return self.observe_state()

    def step(self, action):
        game.clock = pygame.time.get_ticks()
        self.handle_action(action)
        self.update_environment_state()

        if game.alldeadsw and game.chgbg == game.BOSSPO:
            done = True
        elif game.yellow_health <= 0:
            done = True
        else:
            done = False

        game.damage_effect()
        game.handle_bullets()
        game.Monster_movement()
        game.seffect_timer()
        game.change_background()

        self.render()
        # next_state = self.last_minimap_state  # render()에서 생성된 state 사용

        reward = self.calculate_reward()
        state = self.observe_state()
        return state, reward, done, {}
    

    def update_environment_state(self):

        game.yellow_feet = pygame.Rect(game.yellow.x, game.yellow.y + game.yellow.height, game.yellow.width,1) # 주인공 발의 위치 = 발판과 충돌의 정확성위해

        for i in range(len(game.REDS)):                              #몹이 있나? 
            if game.deadcount >= len(game.REDS):                          #죽은 갯수가 몹수랑 같거나 더 많으면 
                game.alldeadsw = True                                #모두 죽었다 스위치 온
          
        # 포탈 충돌
        if game.yellow.colliderect(game.R_portal) and game.alldeadsw:
            self.portal_reached = True
            game.collid_portal()

        # 몬스터와 충돌하면 데미지
        for i, monster in enumerate(game.REDS):
            if game.yellow.colliderect(monster) and game.monswitch[i]:
                game.mydamage_numbers.append(game.DamageNumber(game.yellow.x + game.yellow.width // 2, game.yellow.y,game. MYDMG, game.RED))
                game.ihurt = True
                game.yellow_health -= game.MYDMG
            else:
                game.ihurt = False                
                
        if game.yellow_is_jumping:
            game.yellow.y += game.yellow_y_velocity
            game.yellow_y_velocity += game.GRAVITY
            for plat in game.platforms:
                if game.yellow_feet.colliderect(plat) and game.yellow.centerx >= plat.left <= plat.right and game.yellow_y_velocity > 0:
                    game.on_ground = True
                    game.yellow_is_jumping = False
                    game.yellow.y = plat.top - game.yellow.height            
                             
        if game.yellow.y >= game.GROUND_Y: #땅에 서있을떄 발판에 충돌하지 않을때
            game.yellow.y = game.GROUND_Y                                       #내 위치에 땅의 위치 저장
            game.yellow_is_jumping = False                                 #점프중 스위치 OFF
            game.on_ground = True                                          #땅 스위치 ON
            game.yellow_y_velocity = 0                                     #가속도 0
            
        if not game.on_ground:                                       #떨어지고 있을때 
            game.yellow.y += game.yellow_y_velocity                       #캐릭이 아래로 떨어짐
            game.yellow_y_velocity += game.GRAVITY                        #중력속도만큼 
            

        # 아이템 먹기
        for i, irect in enumerate(game.ITEMrect):
            if game.yellow.colliderect(irect) and game.dropswitch[i]:
                game.get_item(i)
                self.itemget = True
                game.dropswitch[i] = False

    def handle_action(self, action):
        current_time = time.time()
        # delay = self.action_delay_map.get(action, 0.5)  # 기본값 0.5초

        # if current_time - self.action_last_time[action] < delay:
        #     return  # 딜레이 미충족 → 무시

        self.action_last_time[action] = current_time  # 실행 시각 갱신
        """에이전트의 액션을 처리합니다."""
        if action == 0:  # 왼쪽 이동
            self.move_left()
        elif action == 1:  # 오른쪽 이동
            self.move_right()
        elif action == 2:  # 점프
            self.jump()
        elif action == 3:  # 공격
            self.attack()
        elif action == 4:  # 아무것도 하지 않기
            pass
        elif action == 5 and game.skillget == 1:  # 아이템 사용
            game.switch = 1
            game.rect_x = 540
        elif action == 5 and game.skillget == 2:  # 스킬1 사용
            game.switch = 2
            game.rect_x = 610
        elif action == 5 and game.skillget == 3:  # 스킬1 사용
            game.switch = 3
            game.rect_x = 680
        elif action == 5 and game.skillget == 4:  # 스킬2 사용
            game.switch = 4
            game.rect_x = 750
        elif action == 5 and game.skillget == 5:  # 스킬3 사용
            game.switch = 5
            game.rect_x = 820




    def calculate_reward(self):
        reward = 0
        
        if not hasattr(self, 'prev_health'):
            self.prev_health = game.yellow_health
        #체력 상태
        if not hasattr(self, 'half_health_penalized'):
            self.half_health_penalized = False
            
        if not hasattr(self, 'half_health_rewarded'):
            self.half_health_rewarded = True
        
        if game.yellow_health <= 0:
            print(f"dead")
            reward -= 5
        # elif game.yellow_health < self.prev_health:  
        #     reward =- 1  
        if game.yellow_health < game.Maxhealth / 2 and not self.half_health_penalized:
            reward -= 5
            print(f"half health 페널티")
            self.half_health_penalized = True
            self.half_health_rewarded = False  # 다시 보상 받을 수 있도록 초기화
    # elif game.yellow_health > self.prev_health:
    #     reward =+ 10
        if game.yellow_health >= game.Maxhealth / 2 and not self.half_health_rewarded:
            reward += 5
            print(f"full health 복귀")
            self.half_health_rewarded = True
            self.half_health_penalized = False  # 다시 감점 받을 수 있도록 초기화
                
        self.prev_health = game.yellow_health
            
        # 몬스터 처치 보상
        if not hasattr(self, 'prev_monswitch'):
            self.prev_monswitch = game.monswitch.copy()

        for i in range(len(game.REDS)):
            if self.prev_monswitch[i] and not game.monswitch[i]:
                print(f"위치{i} 몬스터 처치 보상")
                reward += 5

        self.prev_monswitch = game.monswitch.copy()
        # 몬스터 피해 입힐시 보상
        if not hasattr(self, 'prev_monster_damage'):
            self.prev_monster_damage = [0] * len(game.REDS)
        for i in range(len(game.REDS)):
            if game.monster_healths[i] and game.monster_healths[i] < self.prev_monster_damage[i]:
                reward += {3: 2, 2: 3, 1: 4}.get(i, 1)

            self.prev_monster_damage[i] = game.monster_healths[i]

        # 플랫폼에 1번 올라가면 보상
        for i,plat in enumerate(game.platforms):
            if game.yellow_feet.colliderect(plat) and game.yellow.centerx >= plat.left <= plat.right and game.yellow_y_velocity and not self.platform_touched[i] and plat != game.platforms[0]:
                self.platform_touched[i] = True
                reward += {3: 15, 2: 20, 1: 20,0: 0}.get(i, 10) 
                print(f"플랫폼에 {i}번 올라가면 보상")

        # 모든 몬스터 처치 시
        if game.alldeadsw:
            print("모든 몬스터 처치")
            reward += 20
            
        if game.alldeadsw and game.BOSSPO == game.chgbg:
            print("보스 처치")
            reward += 20

        # 아이템 수집 보상

        for i in range(len(game.ITEMrect)):
            if game.yellow.colliderect(game.ITEMrect[i]) and game.dropswitch[i]:
                print("아이템 수집")
                reward += 5


        # 포탈 도달 보상
        if hasattr(self, 'portal_reached') and self.portal_reached:
            reward += 10
            print("포탈 도달")
            self.portal_reached = False
            for i in range(len(game.platforms)):
                self.platform_touched[i] = False
        return reward

    def min_max_normalize(self, x, min_val=1000, max_val=70000):
        return (x - min_val) / (max_val - min_val)
    
    def last_reward_timer(self, step_count,MAXCOUNTDOWN=5000):
        self.step_count = step_count
        self.MAXCOUNTDOWN = MAXCOUNTDOWN
        return self.step_count

    def observe_state(self):
        raw_value = game.SKILLDMG[game.switch - 1] * game.critical
        normalized = self.min_max_normalize(raw_value)
        MAX_BULLET_REFERENCE = 100
        bullet_count = len(game.yellow_bullets)
        normalized_bullet_count = math.log(bullet_count + 1) / math.log(MAX_BULLET_REFERENCE + 1)
        state = []
        # 캐릭터의 중심 좌표
        px = game.yellow.x + game.yellow.width / 2
        py = game.yellow.y + game.yellow.height / 2
        self.px = px
        self.py = py

        # total 91차원
        # 1. 캐릭터 위치 (총 10차원)
        state.append(px /game.WIDTH)   # 정규화
        state.append(py / game.HEIGHT)
        # 2. 체력 (정규화)
        state.append(game.yellow_health / game.Maxhealth)
        # 3. 점프
        state.append(1.0 if game.yellow_is_jumping else 0.0)
        # 4. 방향
        state.append(1.0 if game.LRSWITCH == 'r' else 0.0)
        state.append(1.0 if game.on_ground else 0.0)
        # 5. 데미지 충돌
        state.append(1.0 if game.ihurt else 0.0)
        #날라가는 총알갯수
        state.append(normalized_bullet_count)
        # 4. 공력력
        state.append(normalized)  # 정규화
        # 5. 스킬 개수
        state.append(game.skillget/5)       # 예: 총 5개라면 정규화
        
            # 2. 발판 정보 (최대 20 차원)
        max_platforms = 4
        for i in range(max_platforms):
            if i < len(game.platforms):
                plat = game.platforms[i+1]
                bx = plat.x + plat.width / 2
                by = plat.y + plat.height / 2
                dx = (bx - px) / game.WIDTH
                dy = (by - py) / game.HEIGHT
                pdist = ((dx ** 2 + dy ** 2) ** 0.5) /(2 ** 0.5)
                
                state.append(bx / game.WIDTH)
                state.append(by / game.HEIGHT)
                state.append(pdist)
                state.append(1.0 if game.yellow_feet.colliderect(plat) and game.on_ground else 0.0)
                state.append(plat.width / game.WIDTH)
                # state.append(plat.height / game.HEIGHT)

        # 7. countdown
        state.append(self.step_count / self.MAXCOUNTDOWN) 

        # 몬스터 상태 (40차원)
        for i,monster in enumerate(game.REDS):
            if  game.monswitch[i]:
                mx = monster.x + game.MONSTER_WIDTH / 2
                my = monster.y + game.MONSTER_HEIGHT / 2
                dx = (mx - px) / game.WIDTH
                dy = (my - py) / game.HEIGHT
                dist = ((dx ** 2 + dy ** 2) ** 0.5) / (2 ** 0.5)
  
                state.append(mx / game.WIDTH)
                state.append(my / game.HEIGHT)
                state.append(dist)
                state.append(1.0 if game.hit else 0.0)
                state.append(1.0 if game.yellow.colliderect(monster) and game.monswitch[i] else 0.0)
                state.append((game.monster_directions[i]+1)//2)
                state.append(game.MON_VEL / 67)            # 속도 정규화 (최대 10 기준)
                state.append(game.monster_healths[i] / game.MAX_monsterHP[i])
            else:
                state.extend([-1.0,-1.0,-1.0,0.0,0.0,-1.0,0.0,0.0])

        # 아이템 상태 (20차원)
        for i in range(5):
            if game.dropswitch[i]: #떨어진 아이템 스위치가 on 이고 몹이 없을때
                ix = game.itemx[i] + game.ITEM_WIDTH / 2
                iy = game.itemy[i] + game.ITEM_HEIGHT / 2
                dx = (ix - px) / game.WIDTH
                dy = (iy - py) / game.HEIGHT
                dist = ((dx ** 2 + dy ** 2) ** 0.5)/(2 ** 0.5)
                state.append(dist)
                state.append(1.0 if game.dropswitch[i] else 0.0)
                state.append(1.0 if game.yellow.colliderect(game.ITEMrect[i]) and game.dropswitch[i] else 0.0)
                state.append(game.droprd[i] / 5)
            else:
                state.extend([-1.0,0.0,0.0,0.0])

        return np.array(state, dtype=np.float32)
    
    def move_left(self):
        if  game.yellow.x < 0:                                   # 캐릭이 맵밖으로 벗어날떄
            game.yellow.x += game.VEL                                 #캐릭이 벽을 넘지 못하게 반대로 이동
        else: 
            game.yellow.x -= game.VEL                                 #정상일때 왼쪽으로 이동
            game.LRSWITCH = 'l'                                  #좌측 이동 스위치 저장
        
        if game.yellow.colliderect(game.R_portal) and game.alldeadsw:      #포탈에 충돌하는데 몹이 없을때
            game.collid_portal()                                     

        for plat in game.platforms:
            if game.yellow_feet.colliderect(plat):                #캐릭터의 발이 발판에 충돌
                if  plat.left <= game.yellow.centerx:             #발판안에 내 좌표가 있을때 
                    game.on_ground = True                         #땅이라고 알림
                else :
                    game.on_ground = False                        #발판에서 떨어지면 땅이아니라고 알림


    def move_right(self):
        if  game.yellow.x >= game.WIDTH - game.yellow.width:                #캐릭이 우측 끝에 가면
            game.yellow.x -= game.VEL                                  #반대로 움직임
        else:
            game.yellow.x += game.VEL                                  #정상이면 우측으로 움직임
            game.LRSWITCH = 'r'                                   #우측을 본다고 스위치 온
        
        if game.yellow.colliderect(game.R_portal) and game.alldeadsw:       #포탈에 충돌하는데 몹이 없을때
            game.collid_portal()

        for plat in game.platforms:                        
            if game.yellow_feet.colliderect(plat):               #발판에 충돌할때
                if  game.yellow.centerx < plat.right:            #나의 중심위치가 발판의 우측끝보다 안에 있을때
                    game.on_ground = True                        #지면이라고 설정
                else :
                    game.on_ground = False                       #지면이 아니라고 설정

        for i, irect in enumerate(game.ITEMrect):                #아이템과 충돌
            if game.yellow.colliderect(irect) and game.dropswitch[i]: #떨어진 아이템 스위치 on 이면서 아이템과 충돌할때
                game.get_item(i)
                game.dropswitch[i] = False                       #떨어진 아이템 스위치 OFF

    def jump(self):
        if not game.yellow_is_jumping and game.on_ground:                #점프중이 아니고 땅에 서있을때
            if game.clock - game.last_jump_time >= game.JUMP_COOLDOWN:         #점프쿨다운 시간이 지났는지 확인
                game.yellow_is_jumping = True                        #점프중 스위치 ON
                game.yellow_y_velocity = -game.JUMP_POWER                 #점프파워 높이 만큼 가속도에 저장
                game.last_jump_time = game.clock
            
        if game.yellow.colliderect(game.R_portal) and game.alldeadsw:       #포탈에 충돌하는데 몹이 없을때
            game.collid_portal()

 
    def attack(self):
        game.change_skill_image()
        game.shoot_bullet()

    # 미니맵을 그리기 위한 함수

    def render(self):
        if self.render_mode:
            # self.lapclock.tick(30)  # FPS 설정
            game.draw_window()
            
            # # 기존 WIN Surface (900x600)에서 축소된 화면 만들기
            # full_surface = game.WIN.copy()

            # # 축소된 크기로 변환 (예: 1/4 크기)
            # scaled = pygame.transform.scale(full_surface, (225, 150))

            # # 미니맵 surface 생성
            # # self.minimap_surface = pygame.Surface((50, 50)) 
            # screen = pygame.display.get_surface()
            # screen.fill((0, 0, 0))  # 검정 배경
            # screen.blit(scaled, (100, 100))  # 좌상단 작은 화면 출력
            # 예시: 3프레임에 1번만 화면 갱신
            # if self.step_count % 3 == 0:
            #     pygame.display.update()

            
            # 기본 게임화면 렌더링
            # pygame.display.get_surface().blit(game.WIN, (0, 0))

            # 상태 추출 (10x10 사이즈의 간단한 상태)
            # state_vector = extract_state_from_screen(game.WIN,target_size=(10, 10))

            # # 디버그용 minimap 시각화 surface에 그리기
            # draw_minimap_from_state(state_vector, self.minimap_surface, scale=4)

            # # 미니맵 화면에 표시 (오른쪽 위)
            # game.WIN = pygame.display.get_surface()
            # game.WIN.blit(self.minimap_surface, (game.WIN.get_width() -50, 10))

            # pygame.display.update()  # 업데이트 다시 호출해야 반영됨

            # # 화면에서 상태 추출
            # game.WIN = pygame.display.get_surface()
            # state_vector = extract_state_from_screen(game.WIN)  # 84x84 벡터 (flattened)

            # # 상태 벡터를 저장하거나 디버깅용 출력
            # self.last_minimap_state = state_vector  # 원하는 방식으로 저장
            # pygame.display.flip()
