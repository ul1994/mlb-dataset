
sep = '/'
maxEstSize = 1000
CAPTURE_DIR = '_captures'

IMAGE_SIZE = (1024, 1280) # h, w
TOP_PAD, BOT_PAD = 26, 14

active_meshes = [
	666, # arms legs
	766, # body + head
	326, # head
	459, # ?
	704, # arms legs
	306, # torso
	149, # feet
]

BLACKLIST_MESHES = [
	555,
]

# active_meshes += [
#     253, # bat
# ]

class CocoPart:
	Nose = 0
	Neck = 1
	RShoulder = 2
	RElbow = 3
	RWrist = 4
	LShoulder = 5
	LElbow = 6
	LWrist = 7
	RHip = 8
	RKnee = 9
	RAnkle = 10
	LHip = 11
	LKnee = 12
	LAnkle = 13
	REye = 14
	LEye = 15
	REar = 16
	LEar = 17
	Background = 18

def chain(iters):
	ls = []
	for ent in iters:
		if type(ent) != list:
			ent = list(ent)
		ls += ent
	return ls

JOINTS_SPEC = [
	dict(
		name='Head',
		match=[(766, range(0, 220)), (326, range(171, 187))],
	),
	dict(
		name='Neck',
		match=[(766, [612, 625]), (306, [152, 165])]
	),
	dict(
		name='R_Shoulder',
		match=[(766, [722, 717]), (306, [257, 288])]
	),
	dict(
		name='R_Elbow',
		match=[(666, range(90, 104)), (704, [114, 130])],
	),
	dict(
		name='R_Hand',
		match=[(666, range(260, 301)), (704, [250, 260, 313])],
	),
	dict(
		name='L_Shoulder',
		match=[(766, [692, 659]), (306, [199, 232])]
	),
	dict(
		name='L_Elbow',
		match=[(704, [21, 29]), (666, range(35, 50))],
	),
	dict(
		name='L_Hand',
		match=[(666, range(154, 184)), (704, [151, 152, 206])],
	),
	dict(
		name='R_Hip',
		match=[(666, [654, 646]), (704, [654, 646])],
	),
	dict(
		name='R_Knee',
		match=[(666, range(376, 382)), (704, [380, 449])],
	),
	dict(
		name='R_Ankle',
		match=[(149, chain([range(92, 95), range(110, 115)]))],
	),
	dict(
		name='L_Hip',
		match=[(666, [515, 567]), (704, [514, 540])],
	),
	dict(
		name='L_Knee',
		match=[(666, range(575, 581)), (704, [458, 548])],
	),
	dict(
		name='L_Ankle',
		match=[(149, range(45, 50))],
	),
]

CocoColors = [
	[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
	[0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
	[0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

CocoPairs = [
	(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
	(11, 12), (12, 13), (1, 0),
	#(0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19

# CocoPairsRender = CocoPairs[:-2]
