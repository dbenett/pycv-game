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

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
RED = (255,0,0)

HOR_SPEED = 4
VER_SPEED = -14
CHAR_HEIGHT = 30
CHAR_WIDTH = 20

LINE_WIDTH = 2

GRID_WIDTH = 10

WIDTH = 800
HEIGHT = 700

imgfn = sys.argv[1]
bg = pygame.image.load(imgfn)
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))

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
     
    # show the original image and the edge detected image
    cv2.imshow("Image", img)
    cv2.imshow("Edged", edged)

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

    # convert to grayscale
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(imggray, imggray)
    cv2.imshow("eq", cv2.resize(imggray, (0,0), fx=0.2, fy=0.2))
    imggray = cv2.GaussianBlur(imggray, (5,5), 0)

    retval, threshold = cv2.threshold(imggray, 210, 255, cv2.THRESH_BINARY)
    cv2.bitwise_not (threshold, threshold)
    cv2.imshow("thrsh", cv2.resize(threshold, (0,0), fx=0.2, fy=0.2))

    nonzero = cv2.findNonZero(edged)

    #print(nonzero)
    
    #for pt in nonzero:
    #    lines.append(Line(LINE_WIDTH, LINE_WIDTH, pt[0][0], pt[0][1]))

    vlines = []
    hlines = []
    pts = set()
    for pt in nonzero:
        pts.add(snapToGrid(pt[0][0], pt[0][1]))
    for pt in pts:
        hlines.append(Line(LINE_WIDTH, LINE_WIDTH, pt[0], pt[1]))

    #print(sorted(pts))
    '''groups = []
    for k, g in groupby(sorted(pts), key=itemgetter(0)):
        groups.append(list(g)) # Store group iterator as a list
    print(groups)
    for g in groups:
        x = g[0][0]
        g = sorted(g)
        y = g[0][1]
        count = 1
        ind = 1
        while ind < len(g):
            if g[ind][1] == y + count * GRID_WIDTH:
                count += 1
            else:
                height = g[ind-1][1] - y
                newline = Line(LINE_WIDTH, height, x, y)
                vlines.append(newline)
                count = 0
                y = g[ind][1]
            ind += 1
    groups = []
    for k, g in groupby(sorted(pts, key=itemgetter(1)), key=itemgetter(1)):
        groups.append(list(g))
    for g in groups:
        y = g[0][1]
        g = sorted(g)
        x = g[0][0]
        count = 1
        ind = 1
        while ind < len(g):
            if g[ind][0] == x + count * GRID_WIDTH:
                count += 1
            else:
                width = g[ind-1][0] - x
                newline = Line(width, LINE_WIDTH, x, y)
                hlines.append(newline)
                count = 0
                x = g[ind][0]
            ind += 1'''

    return (Player(playerpt[0], playerpt[1]), hlines, vlines)

player, hlines, vlines = generateLevel()

#lines.append(Line(400, LINE_WIDTH, 100, 300))
#lines.append(Line(LINE_WIDTH, 500, 400, 100))

# -------- Main Program Loop -----------
while not quit:
    # --- Main event loop
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

    for hline in hlines:
        hline.draw()    
    for vline in vlines:
        vline.draw()
    
    pygame.display.flip()
    
pygame.quit()
