
from glob import glob
import numpy as np
from configs import *
import cv2
import math, os, sys
from scipy.ndimage import gaussian_filter as blur

def gather_captures(dpath='_captures'):
	framefiles = sorted(
		glob('%s/*.jpg' % dpath),
		key=lambda ent: int(ent.split(sep)[1][6:-4]))
	hashes = list(map(lambda ent: ent.split(sep)[-1][:-4], framefiles))
	return hashes

def gather_meshes():
	markupfiles = sorted(glob('_meshes/*.obj'))
	return markupfiles

def parsev(line, comment=True, dtype=float):
	if comment: line = line.replace('# ', '')
	parts = line.split()
	token = parts[0]
	vlist = [dtype(val) for val in parts[1:]]
	return vlist, token

RSTRIDE = 3
BSTART = 40
def load_markup(fname):
	with open(fname) as fl:
		raw = fl.read()
	lines = [ln for ln in raw.split('\n') if ln]
	registers = lines[1:1+256]
	registers = [parsev(ln)[0] for ln in registers]

	vstart, vend, ventries = -1, -1, 1
	vstart = lines.index('# Data:') + 1
	for lii, line in enumerate(lines[vstart:]):
		if line[0] == 'v':
			ventries = lii + 1
			break
	for lii, line in enumerate(lines[vstart:]):
		if line[0] == 'f':
			vend = lii
			break
	vend += vstart
#     print(vstart, vend, ventries)

	vgrouped = [lines[lii:lii+ventries] for lii in range(vstart, vend, ventries)]
	faces = [parsev(ln, comment=False, dtype=int)[0] for ln in lines if ln[0] == 'f']

	verts = []
	for group in vgrouped:
		ent = {}
		converted = [parsev(line) for line in group[:ventries-1]]
		for vector, token in converted: ent[token] = vector

		vertex, _ = parsev(group[-1], comment=False)
		ent['vertex'] = vertex + [1,]
		verts.append(ent)

	for ent in verts:
		vertex = ent['vertex']
		if 'blend' in ent:
			bind = int(ent['blend'][0])
			assert bind >= 0
			boffset = BSTART + RSTRIDE * bind
			bvect = np.zeros(4)
			for dim in range(3):
				bweights = np.array(registers[boffset+dim])
				bvect[dim] = np.dot(vertex, bweights)
			bvect[3] = 1
			ent['world'] = bvect
		else:
			ent['world'] = ent['vertex']
		ent['project'] = np.array([np.dot(ent['world'], registers[rii]) for rii in range(4)])
		ent['project'] /= ent['project'][-1]

	return verts, faces, registers

def writeobj(fname, verts, faces, component='world'):
	with open(fname, 'w') as fl:
#         print(len(verts))
#         assert False
		for vi, ent in enumerate(verts):
			fl.write('v %s\n' % ' '.join(['%f' % val for val in ent[component][:3]]))
		for fi, fdef in enumerate(faces):

			# if fi < len(faces) - 5:
			# 	fl.write('g i%03d\n' % fi)
			fl.write('f %s\n' % ' '.join(['%d' % val for val in fdef]))

def SameSide(p1, p2, a, b):
    cp1 = np.cross(b-a, p1-a)
    cp2 = np.cross(b-a, p2-a)
    return np.dot(cp1, cp2) >= 0

def inside_triangle(p, a, b, c):
    return SameSide(p, a, b, c) and SameSide(p, b, a, c) \
        and SameSide(p, c, a, b)

def objnum(fname):
	on = int(fname.split(sep)[-1].split('_')[2][1:])
	return on

def frame_meshes(fhash, ls):
	assert len(ls) > 0
	filtered = []
	for ent in ls:
		fmatch = ent.split(sep)[-1].split('_')[0]
		if fhash == fmatch:
			filtered.append(ent)
	filtered = sorted(filtered, key=objnum)
	assert len(filtered) > 0
	return filtered

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 2.0
fontColor              = (255,255,255)
lineType               = 1

def cpos(size, offset):
	return size//2 + offset * size//2

def degenerative(face):
	ls = {}
	for vi in face:
		if vi not in ls:
			ls[vi] = 0
		ls[vi] += 1
	for _, count in ls.items():
		if count > 1:
			return True
	return False

def find_joint(jind, ls):
	for entry in ls:
		if entry['joint_ind'] == jind:
			return entry
	return None
	# return canvas

class mergedstruct:
	def __init__(self):
		self.verts = []
		self.faces = []
		self.foffset = 0
		self.limit = 10

def merge_add(obj, verts, faces):
	if obj.limit == 0:
		return
	else:
		obj.limit -= 1
	foffset = len(obj.verts)
	obj.verts += verts

	offset_faces = []
	for face in faces:
		offset_faces += [[val + foffset for val in face]]
		obj.faces += offset_faces
	assert len(offset_faces) == len(faces)

def load_capture(fhash):
	im = cv2.imread(CAPTURE_DIR + '/%s.jpg' % fhash)
	im = im[TOP_PAD:-BOT_PAD]
	return im

def show_capture(fhash):
	import matplotlib.pyplot as plt
	bg = load_capture(fhash)
	plt.figure(figsize=(8, 8))
	plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
	plt.show();plt.close()

primCount = lambda val: int(val.split('_')[-1].replace('.obj', ''))

def estSize(verts):
	wcoords = np.array([vert['world'] for vert in verts])
	xrange = np.max(wcoords[:, 0]) - np.min(wcoords[:, 0])
	yrange = np.max(wcoords[:, 1]) - np.min(wcoords[:, 1])
	zrange = np.max(wcoords[:, 2]) - np.min(wcoords[:, 2])
	return xrange, yrange, zrange

def depth_order(tris):
	inds = [ei for ei, entry in enumerate(tris)]
	inds = filter(lambda ii: tris[ii]['zdepth'] < 1 and tris[ii]['zdepth'] > 0.8, inds)
	inds = sorted(inds, key=lambda ii: tris[ii]['zdepth'])
	zs = [tris[ii]['zdepth'] for ii in inds]
	# print(zs[0], zs[-1])
	zmin, zmax = min(zs), max(zs)
	return inds, zmin, zmax

def cull_skeletons(skels, bounds):
	hh, ww = bounds
	ls = []
	for person in skels:
		inb = []
		for joint in person:
			xx, yy, zd = joint['pos']
			inb.append((xx >= 0 and xx < ww) and (yy >= 0 and yy < hh))

		if any(inb):
			ls.append(person)
	return ls

def skeleton_bounds(skels):
	boxes = []
	for person in skels:
		xr = [100000000, 0]
		yr = [100000000, 0]
		for joint in person:
			xx, yy, zd = joint['pos']
			if xx < xr[0]: xr[0] = xx
			if xx > xr[1]: xr[1] = xx
			if yy < yr[0]: yr[0] = yy
			if yy > yr[1]: yr[1] = yy
		boxes.append([xr[0], yr[0], xr[1], yr[1]])
	return boxes

def pad_boxes(boxes, ratio=0.15, lim=20):
	padded = []
	for (x0, y0, xf, yf) in boxes:
		hh = yf - y0
		hpad = max(hh * ratio, lim)
		ww = xf - x0
		wpad = max(ww * ratio, lim)
		padded.append([
			x0 - wpad, y0 - hpad,
			xf + wpad, yf + hpad
		])
	return padded

def midpnt(x0, xf):
	return abs(xf - x0) / 2 + min(xf, x0)

def safe_cut(img, cutX, cutY):
	ypad, xpad = [max(-cutY[0], 0), max(-cutX[0], 0)] # y0, x0
	cutImg = img[max(cutY[0], 0):cutY[1], max(cutX[0], 0):cutX[1]]
	return cutImg, xpad, ypad

def blend(points):
	canvas = np.zeros(points.shape[:2])
	for dim in range(len(JOINTS_SPEC)):
		canvas[points[...,dim] > 0] = points[...,dim][points[...,dim] > 0]
	return canvas

ksize = 13
blur_template = np.zeros((ksize, ksize))
blur_template[int(ksize//2), int(ksize//2)] = 1
blur_template = blur(blur_template, 3)
blur_template /= np.max(blur_template)
def place_blur(canvas, xx, yy):
	global blur_template
	ksize = blur_template.shape[0]
	khalf = ksize//2
	y0, x0 = max(yy - khalf, 0), max(xx - khalf, 0)
	ky, kx = max(khalf - yy, 0), max(khalf - xx, 0)
	spot = canvas[y0:y0+ksize-ky, x0:x0+ksize-kx]
	spot[:, :] = blur_template.copy()[ky:ky+spot.shape[0], kx:kx+spot.shape[1]]