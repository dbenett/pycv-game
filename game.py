import pygame
import cv2
import numpy as np
import sys
from itertools import groupby
from operator import itemgetter

if len(sys.argv) < 2:
    print('game.py <levelimg.jpg>')
    sys.exit(2)

pygame.init()
GAMEOVER = "assets/gameover.jpg"
WIN = "assets/win.jpg"
WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
RED = (255,0,0)

HOR_SPEED = 4
VER_SPEED = -14
CHAR_HEIGHT = 15
CHAR_WIDTH = 15
MAX_GRAVITY = CHAR_HEIGHT-1

LINE_WIDTH = 2

GRID_WIDTH = 10

WIDTH = 800
HEIGHT = 700

imgfn = sys.argv[1]
bg = pygame.image.load(imgfn)
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))

go = pygame.image.load(GAMEOVER)
gameover = pygame.transform.scale(go, (WIDTH, HEIGHT))

wi = pygame.image.load(WIN)
win = pygame.transform.scale(wi, (WIDTH, HEIGHT))

screen = pygame.display.set_mode((WIDTH,HEIGHT))

pygame.display.set_caption("Game")

quit = False

arial18 = pygame.font.SysFont('arial',18, False, False)

gameState = 1

class Player():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vY = 0
        self.isJumping = False

    def moveRight(self):
        self.x += HOR_SPEED

    def moveLeft(self):
        self.x -= HOR_SPEED

    def leftCollision(self):
        self.x += HOR_SPEED

    def rightCollision(self):
        self.x -= HOR_SPEED

    def jump(self):
        if self.isJumping == False and self.vY == 0:
            self.vY = VER_SPEED
            self.isJumping = True

    def update(self):
        self.y += self.vY
        if self.vY <= MAX_GRAVITY:
            self.vY += 1

    def getRect(self):
        return pygame.Rect(self.x,self.y,CHAR_WIDTH,CHAR_HEIGHT)

    # manage player collisions. 'other' is of type pygame.Rect
    def stopFall(self, other):
        ground = other.top
        player = self.getRect()
        if self.vY >= 0:
            self.vY = 0
            self.y = ground - CHAR_HEIGHT
            self.isJumping = False

    def draw(self):
        pygame.draw.rect(screen, RED, self.getRect())

    def reset(self):
        self.x = 250
        self.y = 250
        self.yV = 0

class Line():
    def __init__(self, width, height, x, y):
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def getRect(self):
        return pygame.Rect(self.x,self.y,self.width,self.height)

    def isHorizontal(self):
        if self.width > self.height:
            return True

    def draw(self):
        pygame.draw.rect(screen,RED,self.getRect())

    def __str__(self):
        return "w, h, x, y: {}, {}, {}, {}".format(self.width,self.height,self.x,self.y)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return (self.x == other.x and self.y == other.y and self.width == other.width and self.height == other.height)
        return False

def getBiggestWhiteBlob(img):
    _, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('no player! exiting')
        sys.exit(2)
    cnt = max(contours, key=cv2.contourArea)
    return cnt

def snapToGrid(x, y):
    return (int(GRID_WIDTH * round(float(x)/GRID_WIDTH)), int(GRID_WIDTH * round(float(y)/GRID_WIDTH)))

def generateLevel():
    # generate level from image
    img = cv2.imread(imgfn)

    img = cv2.resize(img, (WIDTH, HEIGHT))

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.GaussianBlur(gimg, (5, 5), 0)
    edged = cv2.Canny(gimg, 50, 120)

    imgw, imgh, _ = img.shape

    lower_player = np.array([100, 0, 0])
    upper_player = np.array([255, 100, 100])
    mask = cv2.inRange(img, lower_player, upper_player)
    player = cv2.bitwise_and(img, img, mask= mask)
    pg = cv2.cvtColor(player, cv2.COLOR_BGR2GRAY)
    pgb = cv2.GaussianBlur(pg, (5, 5), 0)
    retval, player = cv2.threshold(pgb, 0, 255, cv2.THRESH_BINARY)
    playercontour = getBiggestWhiteBlob(player)
    playerpt = (snapToGrid(playercontour[0][0][0], playercontour[0][0][1]))
    print ("x, y: {},{}".format(playerpt[0],playerpt[1]))

    lower_win = np.array([0, 100, 0])
    upper_win = np.array([100, 255, 100])
    maske = cv2.inRange(img, lower_win, upper_win)
    end = cv2.bitwise_and(img, img, mask= maske)
    eg = cv2.cvtColor(end, cv2.COLOR_BGR2GRAY)
    egb = cv2.GaussianBlur(eg, (5, 5), 0)
    endretval, end = cv2.threshold(egb, 0, 255, cv2.THRESH_BINARY)
    endarea = getBiggestWhiteBlob(end)
    # playerpt = (snapToGrid(playercontour[0], playercontour[0]))
    end=[]
    for e in endarea:
        end.append(e[0])

    nonzero = cv2.findNonZero(edged)


    vlines = []
    hlines = []
    pts = set()
    for pt in nonzero:
        pts.add(snapToGrid(pt[0][0], pt[0][1]))
    for pt in pts:
        hlines.append(Line(LINE_WIDTH, LINE_WIDTH, pt[0], pt[1]))

    groups = []
    for k, g in groupby(sorted(pts), key=itemgetter(0)):
        groups.append(list(g)) # Store group iterator as a list
    for g in groups:
        g = sorted(g, key=itemgetter(1))
        prev_x = g[0][0]
        prev_y = g[0][1]
        for g_i in g[1:]:
            if abs(prev_y - g_i[1]) <= GRID_WIDTH:
                vlines.append(Line(LINE_WIDTH, LINE_WIDTH, g_i[0], g_i[1]))
                hlines.remove(Line(LINE_WIDTH, LINE_WIDTH, g_i[0], g_i[1]))
            prev_y = g_i[1]

    return (Player(playerpt[0], playerpt[1]), hlines, vlines, end)

player, hlines, vlines, end = generateLevel()
player_orig = (player.x, player.y)

# -------- Main Program Loop -----------
def play():
    win = False
    player.x = player_orig[0]
    player.y = player_orig[1]
    play=True
    while(play):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_LEFT]:
            player.moveLeft()
        if pressed[pygame.K_RIGHT]:
            player.moveRight()
        if pressed[pygame.K_UP]:
            player.jump()
        #screen.fill(WHITE)

        screen.blit(bg, [0, 0])

        player.update()
        for hline in hlines:
            if hline.getRect().colliderect(player.getRect()):
                player.stopFall(hline.getRect())

        for vline in vlines:
            if vline.getRect().colliderect(player.getRect()):
                lcollide = abs(vline.getRect().right - player.getRect().left)
                rcollide = abs(vline.getRect().left - player.getRect().right)
                if lcollide < rcollide:
                    player.leftCollision()
                else:
                    player.rightCollision()
        player.draw()

        if player.getRect().top > HEIGHT:
            play=False
        for e in end:
            if player.getRect().collidepoint(e):
                play=False
                win=True
        pygame.display.flip()
    return win
w = play()
while not quit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
    # --- Main event loop
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_r]:
        w = play()
    if pressed[pygame.K_q]:
        quit=True
    if w:
        screen.blit(win, [0, 0])
        pygame.display.flip()
    else:
        screen.blit(gameover, [0, 0])
        pygame.display.flip()


pygame.quit()
