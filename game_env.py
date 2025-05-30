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
        self.target_id = 0
        self.prev_threat = False
        self.prev_monster_directions = [0] * 5
        self.lapclock = pygame.time.Clock()
        self.v_threat = False  # 수직 위협 여부
        self.h_threat = False  # 수평 위협 여부
        self.MAXCOUNTDOWN = 10000  # 또는 원하는 초기값
        # self.minimap_surface = pygame.Surface((40, 40))  # 확대 미니맵 Surface
        self.action_last_time = [0.0] * 10  # 각 액션별 마지막 실행 시각
        self.action_delay_map = {
            0: 0.0,   # 왼쪽 이동
            1: 0.0,   # 오른쪽 이동
            2: 0.5,    # 점프
            3: 0.3,    # 공격
            4: 0.0,    # 대기 (즉시 허용)
            5: 5.0     # 스킬1
        }
        game.init_game()
        game.init_assets()

    def reset(self):
        game.init_game()
        game.init_assets()
        self.itemget = False
        self.last_shaping_reward = 0.0
        self.half_health_rewarded = True
        self.half_health_penalized = False  # 다시 감점 받을 수 있도록 초기화
        self.target_id = 0  # 목표 단계 초기화
        self.v_threat = False
        self.h_threat = False
        self.prev_threat = False
        self.prev_idist = [800] * 5
        self.prev_podist = 800
        self.prev_health = 0
        self.prev_damage = 0
        self.was_close = [False for _ in range(len(game.REDS))]
        self.target_platform_order = [4, 3, 2, 1]
        
        self.platform_dx_before_jump = [None] * len(game.platforms)
        # self.last_jump_target_dx = [None] * len(game.platforms)
        self.platform_touched = [False] * len(game.platforms)

        self.prev_pdist = [400] * 5
        self.prev_monster_collision = [False for _ in range(len(game.REDS))]

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
        
        state, target_onehot = self.observe_state()
        reward = self.calculate_reward()
        return state, target_onehot, reward, done, {}
    

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
                           
        # 아이템 먹기
        for i, irect in enumerate(game.ITEMrect):
            if game.yellow.colliderect(irect) and game.dropswitch[i]:
                game.get_item(i)
                self.itemget = True
                game.dropswitch[i] = False
                
        if game.yellow_is_jumping:
            game.yellow.y += game.yellow_y_velocity
            game.yellow_y_velocity += game.GRAVITY
            for i,plat in enumerate(game.platforms):
                if game.yellow_feet.colliderect(plat) and game.yellow.centerx >= plat.left <= plat.right and game.yellow_y_velocity > 0:
                    game.on_ground = True
                    game.on_platform[i] = True
                    game.yellow_is_jumping = False
                    game.yellow.y = plat.top - game.yellow.height            
                             
        if game.yellow.y >= game.GROUND_Y:                              #땅에 서있을떄 발판에 충돌하지 않을때
            game.yellow.y = game.GROUND_Y                                       #내 위치에 땅의 위치 저장
            game.yellow_is_jumping = False                                 #점프중 스위치 OFF
            game.on_ground = True                                          #땅 스위치 ON
            game.yellow_y_velocity = 0                                     #가속도 0
            
        if not game.on_ground:                                       #떨어지고 있을때 
            game.yellow.y += game.yellow_y_velocity                       #캐릭이 아래로 떨어짐
            game.yellow_y_velocity += game.GRAVITY                        #중력속도만큼 
            


    def handle_action(self, action):
        current_time = time.time()
        delay = self.action_delay_map.get(action, 0.0)  # 기본값 0.0초

        if current_time - self.action_last_time[action] < delay:
            return  # 딜레이 미충족 → 무시

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

    def predict_future_collision(self, monster, monster_direction, frames=30):
        y0 = game.yellow.centery
        v0 = game.yellow_y_velocity
        a = game.GRAVITY
        x0 = game.yellow.centerx
        vx = game.VEL if game.LRSWITCH == 'r' else -game.VEL

        m_x0 = monster.centerx
        m_vx = game.monster_speed * monster_direction  # ← 속도 반영
        m_y = monster.centery
        m_h = game.MONSTER_HEIGHT
        m_w = game.MONSTER_WIDTH
        y_h = game.yellow.height
        y_w = game.yellow.width

        for t in range(frames):
            yt = y0 + v0 * t + 0.5 * a * t ** 2
            xt = x0 + vx * t
            mxt = m_x0 + m_vx * t

            # 충돌 조건: x와 y 모두 겹침
            if abs(xt - mxt) < (m_w + y_w) / 2 and abs(yt - m_y) < (m_h + y_h) / 2:
                return True  # 예측 충돌
        return False



    def calculate_reward(self):
        reward = 0

        reward += self.last_shaping_reward # 몬스터 방향 공격 데미지
        
        if not self.prev_damage:
            self.prev_damage = self.raw_value
        
        if self.prev_damage < self.raw_value :
            reward += 0.1
            print(f"damage up bonus raw_value={self.raw_value}")
            self.prev_damage = self.raw_value
        #체력
        if not hasattr(self, 'prev_health'):
            self.prev_health = game.yellow_health
        #체력 상태
        if not hasattr(self, 'half_health_penalized'):
            self.half_health_penalized = False
            
        if not hasattr(self, 'half_health_rewarded'):
            self.half_health_rewarded = True
        
        if game.yellow_health <= 0:
            print(f"dead")
            reward -= 1

        if game.yellow_health < game.Maxhealth / 2 and not self.half_health_penalized:
            reward -= 0.8
            print(f"half health 페널티 game.yellow_health={game.yellow_health}")
            self.half_health_penalized = True
            self.half_health_rewarded = False  # 다시 보상 받을 수 있도록 초기화

        if game.yellow_health >= game.Maxhealth / 2 and not self.half_health_rewarded:
            reward += 0.8
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
                reward += {3: 0.6, 2: 0.8, 1: 1}.get(i, 0.5)
            
                
                
        self.prev_monswitch = game.monswitch.copy()
        # 몬스터 피해 입힐시 보상
        if not hasattr(self, 'prev_monster_damage'):
            self.prev_monster_damage = [0] * len(game.REDS)
            
        for i,monster in enumerate(game.REDS):

            if not game.monswitch[i]:
                continue
            # elif game.monster_healths[i] and game.monster_healths[i] < self.prev_monster_damage[i]:
            #     reward += {3: 0.02, 2: 0.03, 1: 0.04}.get(i, 0.01)
            dx = abs(monster.centerx - game.yellow.centerx)
            dy = abs(monster.centery - game.yellow.centery)
                
            self.prev_monster_damage[i] = game.monster_healths[i]
            
            if 3>i>0:

                FRAMES_TO_COLLIDE = 16

                jump_top = game.yellow.top - 130  # 머리 위치

                # Y축 충돌 범위
                y_threat = not (jump_top > monster.bottom)
 
                approaching = (
                    monster.centerx < game.yellow.centerx and game.monster_directions[i] == 1 or
                    monster.centerx > game.yellow.centerx and game.monster_directions[i] == -1
                    )
                away = (
                    monster.centerx < game.yellow.centerx and game.monster_directions[i] == -1 or
                    monster.centerx > game.yellow.centerx and game.monster_directions[i] == 1
                )

                
                max_jump_x_range = (FRAMES_TO_COLLIDE * game.monster_speed) + 100
                
                x_threat =  approaching and dx < max_jump_x_range
                x_awaythreat = away and dx > max_jump_x_range
                
                if y_threat and (x_threat or x_awaythreat) and not game.yellow_is_jumping:
                    self.v_threat = True
                    # print(f"Danger dx: {dx}, dy: {abs(dy)}")
                elif not x_threat and not x_awaythreat and not game.yellow_is_jumping:
                    self.v_threat = False
                    # print(f"no Danger  dx: {dx}, dy: {abs(dy)}")

                if self.prev_monster_directions[i] != game.monster_directions[i] and self.v_threat:
                    if not game.yellow_is_jumping: 
                    # 위험 조건 모두 만족 & 점프 안했을 때
                        print(f"몹이 위에 NO! jump dx: {abs(dx)},dy: {abs(dy)}, v_threat: {self.v_threat}")
                        reward += 0.5  # 점프 안해서 위험을 회피함
                    else:
                        print(f"몹이 위에 jump dx: {abs(dx)},dy: {abs(dy)}, v_threat: {self.v_threat}")
                        reward -= 0.5
                
                    self.prev_monster_directions[i] = game.monster_directions[i]
                    
                if y_threat and game.yellow_is_jumping:
                    
                    is_danger_predicted = self.predict_future_collision(monster, game.monster_directions[i], frames=30)
                    if is_danger_predicted:
                        self.v_threat = True
                        # reward -= 0.1
                    else:
                        self.v_threat = False
                        # reward += 0.1
                        
                self.prev_threat = self.v_threat
            collided = game.yellow.colliderect(monster)
            # 충돌전에 피했을때 보너스    
               
            if dx < 160 and dy< 50 and not collided and not self.was_close[i]:
                self.was_close[i] = True  # 위협 상황 발생
                print(f"충돌 위험 ! dx: {dx}, dy: {dy}")
                self.h_threat = True
                
                
            elif self.was_close[i] and dx > 200 and dy < 50:
                reward += 0.5  # 회피 성공 보상
                print(f"충돌전에 피했을때 dx: {dx}, dy: {dy}")
                self.h_threat = False
                self.was_close[i] = False  # 보상은 1번만
                
            if collided and not self.prev_monster_collision[i]:
                reward -= 0.8  # 페널티 부여
                self.prev_monster_collision[i] = True
                print(f"몬스터 {i}와 충돌,dx: {dx}, dy: {dy},v_threat: {self.v_threat}, x_threat: {self.h_threat}, hp: {game.yellow_health}")
            elif not collided:
                self.prev_monster_collision[i] = False
                
            # ... threat 계산 후
            if self.v_threat != self.prev_threat:
                print(f"[DEBUG] dx = {dx}, dy = {dy}, v_threat changed: {self.prev_threat} -> {self.v_threat}")


        # # 목표 발판 순서 for platforms

        # platform_order = [0, 4, 3, 2, 1]

        # for idx in range(len(platform_order) - 1):
        #     cur = platform_order[idx]
        #     nxt = platform_order[idx + 1]

        #     if self.platform_touched[cur] and not self.platform_touched[nxt]:
        #         plat = game.platforms[nxt]

        #         dx_left = abs(game.yellow.right - plat.left)
        #         dx_right = abs(game.yellow.left - plat.right)
        #         dy = plat.top - game.yellow.bottom

        #         horizontal_reach = 150
        #         vertical_reach = 105
        #         distance_threshold = 60

        #         if (min(dx_left, dx_right) <= horizontal_reach) and (0 < dy <= vertical_reach):
        #             # 목표 플랫폼 중심까지 거리
        #             dx = abs(plat.centerx - game.yellow.centerx)

        #             # 가장 가까운 방향으로 점프해야함
        #             closer_edge = 'left' if dx_left < dx_right else 'right'
        #             char_x = game.yellow.left if closer_edge == 'left' else game.yellow.right
        #             target_x = plat.left if closer_edge == 'left' else plat.right
        #             edge_dx = abs(target_x - char_x)

        #             # 점프 시작
        #             if game.yellow_is_jumping and self.platform_dx_before_jump[nxt] is None:
        #                 self.platform_dx_before_jump[nxt] = edge_dx

        #             # 점프 종료
        #             if not game.yellow_is_jumping and self.platform_dx_before_jump[nxt] is not None:
        #                 dx_before = self.platform_dx_before_jump[nxt]
        #                 dx_after = edge_dx
        #                 self.platform_dx_before_jump[nxt] = None

        #                 if dx_after < dx_before - distance_threshold:
        #                     reward += 0.3
        #                     print(f"플랫폼 {cur}→{nxt} 방향 점프 보상, edge_dx_before:{dx_before}, after:{dx_after}")

        # # 점프 중 발판 접근 (보조 보상)
        # # (2) 플랫폼 순서에 따라 다음 타겟 정하기
        # # 순서: 0 → 4 → 3 → 2 → 1
        # platform_order = [0, 4, 3, 2, 1]

        # for idx in range(len(platform_order) - 1):
        #     cur = platform_order[idx]
        #     nxt = platform_order[idx + 1]

        #     if self.platform_touched[cur] and not self.platform_touched[nxt]:
        #         # 캐릭터가 다음 플랫폼(nxt)에 얼마나 가까운지 계산
        #         plat = game.platforms[nxt]
        #         px = plat.centerx - game.yellow.centerx
        #         py = plat.centery - game.yellow.centery
        #         pdist = (px ** 2 + py ** 2) ** 0.5

        #         if py < 40:  # 너무 위에 있거나 아래서 점프 중이면 보상 제외
        #             if self.prev_pdist[nxt] - pdist > 150:
        #                 reward += 0.2
        #                 print(f"🔼 {cur}→{nxt} 접근 보상 pdist: {pdist:.1f}")
        #             elif self.prev_pdist[nxt] - pdist < -150:
        #                 reward -= 0.2
        #                 print(f"🔽 {cur}→{nxt} 멀어짐 패널티 pdist: {pdist:.1f}")
        #             self.prev_pdist[nxt] = pdist


        # 점프 전에 거리 저장
        if not hasattr(self, 'platform_dx_before_jump'):
            self.platform_dx_before_jump = [None] * len(game.platforms)

        # 플랫폼에 1번 올라가면 보상
        for i,plat in enumerate(game.platforms):
            if game.on_platform[i] and not self.platform_touched[i]:
                self.platform_touched[i] = True
                # reward += {3: 0.60, 2: 0.80, 1: 0.8,0: 0.0}.get(i, 1.0) 
                reward += {0: 0.0}.get(i, 1.0) 
                print(f"플랫폼에 {i}번 올라가면 보상")
                game.on_platform[i] = False
            else:
                dx_left = abs(game.yellow.right - plat.left)
                dx_right = abs(game.yellow.left - plat.right)
                dy = plat.top - game.yellow.bottom
                horizontal_reach = 150
                vertical_reach = 105
                distance_threshold = 60
                
                if (min(dx_left, dx_right) <= horizontal_reach) and (0 < dy <= vertical_reach):
                    # 점프 시작할 때, 거리 기록
                    if game.yellow_is_jumping and self.platform_dx_before_jump[i] is None:
                        self.platform_dx_before_jump[i] = abs(plat.centerx - game.yellow.centerx)

                    # 점프 끝났을 때 거리 비교
                    if not game.yellow_is_jumping and self.platform_dx_before_jump[i] is not None:
                        dx_after = abs(plat.centerx - game.yellow.centerx)
                        dx_before = self.platform_dx_before_jump[i]
                        self.platform_dx_before_jump[i] = None

                        # 플랫폼 끝 방향으로 점프하며 가까워질 때
                        if dx_after < dx_before - distance_threshold:
                            reward += 0.7
                            print(f"플랫폼 {i}에 점프 dx_before:{dx_before}, dx_after:{dx_after}")
                            # game.on_platform[i] = True
            if i != 4:
                j = i
                px = plat.centerx - game.yellow.centerx  # x축 거리만 사용
                py = plat.centery - game.yellow.centery
                pdist = math.sqrt(px ** 2 + py ** 2)

                if self.platform_touched[j+1] and not self.platform_touched[j] and j != 0:
                    if self.prev_pdist[j] - pdist > 150 and py < 40:
                        reward += 0.4
                        print(f"come 플랫폼 {j} px:{self.px},Py:{self.py},Pdist:{pdist:.1f},self.prev_dist[i]:{self.prev_pdist[j]:.1f}")
                        self.prev_pdist[j] = pdist
                    elif self.prev_pdist[j] - pdist < -150 and py < 40:
                        reward -= 0.4
                        print(f"far 플랫폼 {j} px:{self.px},Py:{self.py},Pdist:{pdist:.1f},self.prev_dist[i]:{self.prev_pdist[j]:.1f}")
                        self.prev_pdist[j] = pdist
                elif j == 0:
                    j = 4
                    if not self.platform_touched[j] and self.prev_pdist[j] - pdist > 150:
                        reward += 0.4
                        print(f"come 플랫폼 {j} px:{self.px},Py:{self.py},Pdist:{pdist:.1f},self.prev_dist[j]:{self.prev_pdist[j]:.1f}")
                        self.prev_pdist[j] = pdist
                    elif not self.platform_touched[j] and self.prev_pdist[j] - pdist < -150:
                        reward -= 0.4
                        print(f"far 플랫폼 {j} px:{self.px},Py:{self.py},Pdist:{pdist:.1f},self.prev_dist[j]:{self.prev_pdist[j]:.1f}")
                        self.prev_pdist[j] = pdist
                    j = 0

        # 모든 몬스터 처치 시
        if game.alldeadsw:
            podistx = game.R_portal.centerx - game.yellow.centerx
            podisty = game.R_portal.centery - game.yellow.centery
            podist = math.sqrt(podistx ** 2 + podisty ** 2)
            if self.prev_podist - podist > 150:
                reward += 0.6
                print(f"Come 포탈 {podist:.1f},self.prev_dist[i]:{self.prev_podist:.1f}")
                self.prev_podist = podist
            if self.prev_podist - podist < -150 :
                reward -= 0.6
                print(f"Far 포탈 {podist:.1f},self.prev_dist[i]:{self.prev_podist:.1f}")
                self.prev_podist = podist
            
        if game.alldeadsw and game.BOSSPO == game.chgbg:
            print("보스 처치")
            reward += 1
        
        for i in range(len(game.ITEMrect)):
            if game.dropswitch[i]:
                ix = game.itemx[i] + game.ITEM_WIDTH / 2
                iy = game.itemy[i] + game.ITEM_HEIGHT / 2
                dx = ix - self.px
                dy = iy - self.py
                idist = math.sqrt(dx ** 2 + dy ** 2)
                if self.prev_idist[i] - idist > 150:
                    reward += 0.4
                    print(f"Come 아이템 {i} px:{self.px},Py:{self.py},idist:{idist:.1f},self.prev_dist[i]:{self.prev_idist[i]:.1f}")
                    self.prev_idist[i] = idist
                if self.prev_idist[i] - idist < -150:
                    reward -= 0.4
                    print(f"Far 아이템 {i} px:{self.px},Py:{self.py},idist:{idist:.1f},self.prev_dist[i]:{self.prev_idist[i]:.1f}")
                    self.prev_idist[i] = idist


        # 아이템 수집 보상
        if self.itemget:  # 아이템 수집 시
            print("아이템 수집")
            reward += 0.5
            self.itemget = False

        # for i in range(len(game.ITEMrect)):
        #     if game.yellow.colliderect(game.ITEMrect[i]) and game.dropswitch[i]:
        #         print("아이템 수집")
        #         reward += 5

  
        # 포탈 도달 보상
        if hasattr(self, 'portal_reached') and self.portal_reached and game.alldeadsw:
            reward += 1
            print("포탈 도달")
            self.portal_reached = False
            for i in range(len(game.platforms)):
                self.platform_touched[i] = False
                
        # (추가) target_id 목표 달성 여부 확인 및 단계 전환
        if self.target_id == 0:
            if not game.monswitch[0] and not game.monswitch[4]:
                self.target_id += 1
                reward += 1.0
                print("목표 0 완료")
        elif self.target_id == 1:
            if game.on_platform[4]:
                self.target_id += 1
                reward += 1.0
                print("목표 1 완료")
        elif self.target_id == 2:
            if not game.monswitch[3]:
                self.target_id += 1
                reward += 1.0
                print("목표 2 완료")
        elif self.target_id == 3:
            if game.on_platform[3]:
                self.target_id += 1
                reward += 1.0
                print("목표 3 완료")
        elif self.target_id == 4:
            if not game.monswitch[2]:
                self.target_id += 1
                reward += 1.0
                print("목표 4 완료")
        elif self.target_id == 5:
            if game.on_platform[2]:
                self.target_id += 1
                reward += 1.0
                print("목표 5 완료")
        elif self.target_id == 6:
            if not game.monswitch[1]:
                self.target_id += 1
                reward += 1.0
                print("목표 6 완료")
        elif self.target_id == 7:
            if game.yellow.colliderect(game.R_portal) and game.alldeadsw:
                self.target_id = 0  # 다음 판으로 리셋
                reward += 1.5  # 전체 목표 완수 보상
                print("목표 7 완료, 다음 판으로 리셋")

                
        return reward

    def min_max_normalize(self, x, min_val=1000, max_val=70000):
        return (x - min_val) / (max_val - min_val)
    
    def last_reward_timer(self, step_count,MAXCOUNTDOWN=5000):
        self.step_count = step_count
        self.MAXCOUNTDOWN = MAXCOUNTDOWN
        return self.step_count

    def observe_state(self):
        self.raw_value = game.SKILLDMG[game.switch - 1] * game.critical
        normalized = self.min_max_normalize(self.raw_value)
        # MAX_BULLET_REFERENCE = 100
        # bullet_count = len(game.yellow_bullets)
        # normalized_bullet_count = math.log(bullet_count + 1) / math.log(MAX_BULLET_REFERENCE + 1)
        state = []
        # 캐릭터의 중심 좌표
        px = game.yellow.centerx
        py = game.yellow.centery
        self.px = px
        self.py = py

        # total 93차원 -> 28차원 -> +8 36차원
        # 1. 캐릭터 위치 (총 9차원)
        state.append(self.px /game.WIDTH)   # 정규화
        state.append(self.py / game.HEIGHT)
        # 2. 체력 (정규화)
        state.append(game.yellow_health / game.Maxhealth)
        # 3. 점프
        # state.append(1.0 if game.yellow_is_jumping else 0.0)
        state.append(1.0 if game.on_ground else 0.0)
        # 4. 방향
        state.append(1.0 if game.LRSWITCH == 'r' else 0.0)
        # 5. 데미지 충돌
        state.append(1.0 if game.ihurt else 0.0)
        
        state.append(1.0 if self.v_threat else 0.0)  # 몬스터 수직 위협 여부
        state.append(1.0 if self.h_threat else 0.0)  # 몬스터 수평 위협 여부
        #날라가는 총알갯수
        # state.append(normalized_bullet_count)
        # 4. 공력력
        state.append(normalized)  # 정규화
        # 5. 스킬 개수
        # state.append(game.skillget/5)       # 예: 총 5개라면 정규화
        
        # 2. 발판 정보 (최대 8차원: 거리 + 접촉 여부 × 4개)
        platform_info = []

        # platforms[1]부터 시작 (platforms[0]은 시작 발판)
        for plat in game.platforms[1:]:  
            bx = plat.x + plat.width / 2
            by = plat.y + plat.height / 2
            dx = (bx - self.px) / game.WIDTH
            dy = (by - self.py) / game.HEIGHT
            pdist = math.sqrt(dx ** 2 + dy ** 2)
            contact = 1.0 if game.yellow_feet.colliderect(plat) and game.on_ground else 0.0
            platform_info.append((pdist, contact))

        # 거리 기준으로 정렬 (먼 발판이 먼저)
        platform_info.sort(reverse=True, key=lambda x: x[0])

        # 4 -> 3 -> 2 -> 1 순서로 state에 저장
        max_platforms = 4
        for i in range(max_platforms):
            if i < len(platform_info):
                pdist, contact = platform_info[i]
                state.append(pdist)
                state.append(contact)
            else:
                state.append(0.0)
                state.append(0.0)



        # 7. countdown 시간제한 3000초 reward시 초기화
        state.append(self.step_count / self.MAXCOUNTDOWN) 


        # 몬스터 2마리 상태 (각 2개 특성씩, 총 4개)
        track_pairs = [(0, 4), (4, 3), (3, 2), (2, 1)]

        if not hasattr(self, 'pair_index'):
            self.pair_index = 0

        current_pair = track_pairs[self.pair_index % len(track_pairs)]

        for mi in current_pair:
            if game.monswitch[mi]:
                mx = game.REDS[mi].x + game.MONSTER_WIDTH / 2
                my = game.REDS[mi].y + game.MONSTER_HEIGHT / 2
                dx = (mx - self.px) / game.WIDTH
                dy = (my - self.py) / game.HEIGHT
                mdist = math.sqrt(dx ** 2 + dy ** 2)

                state.append(mdist)
                # state.append(1.0 if game.hit else 0.0)
                # state.append(1.0 if game.yellow.colliderect(game.REDS[mi]) else 0.0)
                state.append(game.monster_healths[mi] / game.MAX_monsterHP[mi])
            else:
                # 추적 대상 아님 또는 사망시 dummy 값
                state.append(-1.0)
                # state.append(-1.0)
                # state.append(0.0)
                state.append(0.0)

        # 몬스터 중 한 마리라도 죽으면 다음 쌍으로 넘어감
        monsters_alive = [game.monswitch[i] and game.monster_healths[i] > 0 for i in current_pair]
        if sum(monsters_alive) < len(current_pair):
            self.pair_index += 1



        # 아이템 상태 (5차원)
        for i in range(5):
            if game.dropswitch[i]: #떨어진 아이템 스위치가 on 이고 몹이 없을때
                ix = game.itemx[i] + game.ITEM_WIDTH / 2
                iy = game.itemy[i] + game.ITEM_HEIGHT / 2
                dx = (ix - px) / game.WIDTH
                dy = (iy - py) / game.HEIGHT
                dist = math.sqrt(dx ** 2 + dy ** 2)
                state.append(dist)
                # state.append(1.0 if game.dropswitch[i] else 0.0)
                # state.append(1.0 if game.yellow.colliderect(game.ITEMrect[i]) and game.dropswitch[i] else 0.0)
                # state.append(game.droprd[i] / 5)
            else:
                # state.extend([-1.0,0.0,0.0,0.0])
                state.append(-1.0)
        
        #포탈 상태 (2차원)
        #포탈 생성 상태
        # state.append(1.0 if game.alldeadsw else 0.0)
        #포탈 충돌 상태
        # state.append(1.0 if game.yellow.colliderect(game.R_portal) and game.alldeadsw else 0.0)
        
        #포탈 상태 (1차원)
        if game.alldeadsw:
            podistx = game.R_portal.centerx - game.yellow.centerx
            podisty = game.R_portal.centery - game.yellow.centery
            podist = math.sqrt(podistx ** 2 + podisty ** 2)
        state.append(podist if game.alldeadsw else -1.0)

        # observe_state 내부, state 생성 마지막에 아래 추가 8차원
        target_onehot = [0.0] * 8
        target_onehot[self.target_id] = 1.0

        return np.array(state, dtype=np.float32), np.array(target_onehot, dtype=np.float32)
    
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

    def jump(self):
        if not game.yellow_is_jumping and game.on_ground:                #점프중이 아니고 땅에 서있을때
            if game.clock - game.last_jump_time >= game.JUMP_COOLDOWN:         #점프쿨다운 시간이 지났는지 확인
                game.yellow_is_jumping = True                        #점프중 스위치 ON
                game.yellow_y_velocity = -game.JUMP_POWER                 #점프파워 높이 만큼 가속도에 저장
                game.last_jump_time = game.clock
            
        if game.yellow.colliderect(game.R_portal) and game.alldeadsw:       #포탈에 충돌하는데 몹이 없을때
            game.collid_portal()

 
    def attack(self):
        game.change_skill_image()  # 스킬 이미지 변경
        game.shoot_bullet()        # 총알 방향 설정
        self.shaping_reward_for_attack()


    def shaping_reward_for_attack(self):
        # 원하는 추적 순서
        attack_order = [0, 4, 3, 2, 1]

        for idx in attack_order:
            if game.monswitch[idx]:  # 살아있는 몬스터만
                monster = game.REDS[idx]
                dy = abs(monster.centery - game.yellow.centery)
                # 오른쪽 몬스터 + 오른쪽 방향
                if monster.centerx > game.yellow.centerx and dy < 40 and game.LRSWITCH == 'r':
                    self.last_shaping_reward = 0.1
                    return
                # 왼쪽 몬스터 + 왼쪽 방향
                elif monster.centerx < game.yellow.centerx and dy < 40 and game.LRSWITCH != 'r':
                    self.last_shaping_reward = 0.1
                    return

        # 어떤 몬스터도 보너스 조건 만족 못하면 0
        self.last_shaping_reward = 0.0


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
