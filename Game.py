from ple import PLE
import numpy as np
from ple.games import pixelcopter
import datetime as dt
import scipy.misc
import CopterAgent
import DeepTNet

game=pixelcopter.Pixelcopter(width=512,height=512)
network=DeepTNet.DeepTNet()
fps = 30 #fps we want to run at
frame_skip = 2
num_steps = 1
force_fps = False #slower speed
display_screen = True

reward = 0.0
max_noops = 20
nb_frames = 1000

p = PLE(game, fps=fps, frame_skip=frame_skip, num_steps=num_steps,
	force_fps=force_fps, display_screen=display_screen)

p.init()
agent = CopterAgent.CopterAgent(plenv=p,game=game,Qnetwork=network)
noAction=p.NOOP
agent.sampleRandomRun(13)
agent.play()