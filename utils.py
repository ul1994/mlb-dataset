
from glob import glob
import numpy as np
from configs import *
import cv2
import math, os, sys
from scipy.ndimage import gaussian_filter as blur
from scipy.ndimage import uniform_filter as clrblur

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
	on = fname.split(sep)[-1].split('_')[2][1:]
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
fontScale              = 0.5
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

def show_capture(fhash):
	import matplotlib.pyplot as plt
	bg = cv2.imread(CAPTURE_DIR + '/%s.jpg' % fhash)
	bg = bg[30:-10]
	plt.figure(figsize=(8, 8))
	plt.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
	plt.show();plt.close()

def draw_joints(fhash, tag, kpoints, spec=JOINTS_SPEC, colored=True):
	import matplotlib.pyplot as plt

	bg = cv2.imread(CAPTURE_DIR + '/%s.jpg' % fhash)
	bg = bg.astype(np.float32) * 0.25
	bg = bg[30:-10]
	imsize = bg.shape[:2]

	kpoints_overlay = np.zeros(imsize + (3,))
	text_layer = np.zeros(imsize + (3,), dtype=np.uint8)

	for kii, kp in enumerate(kpoints):
		joint_ind, (xx, yy, zz) = kp['joint_ind'], kp['pos']
		if yy >= imsize[0] or yy < 0 or xx >= imsize[1] or xx < 0:
			continue

		clr = CocoColors[joint_ind] if colored else [255, 255, 255]
		kpoints_overlay[int(yy), int(xx), :] = clr

		cv2.putText(text_layer, spec[joint_ind]['name'].lower(),
			(int(xx), int(yy)),
			font,
			fontScale,
			clr,
			lineType)

	kpoints_overlay = clrblur(kpoints_overlay, size=(6, 6, 1))
	kpoints_overlay /= np.max(kpoints_overlay)

	bg += kpoints_overlay * 255 * 0.75
	bg = bg * 0.75 + text_layer.astype(np.float32) * 0.25
	bg = bg.astype(np.uint8)
	cv2.imwrite('_outputs/%s_%s.png' % (fhash, tag), bg)

pcount = lambda val: int(val.split('_')[-1].replace('.obj', ''))

def estSize(verts):
	wcoords = np.array([vert['world'] for vert in verts])
	xrange = np.max(wcoords[:, 0]) - np.min(wcoords[:, 0])
	yrange = np.max(wcoords[:, 1]) - np.min(wcoords[:, 1])
	zrange = np.max(wcoords[:, 2]) - np.min(wcoords[:, 2])
	return xrange, yrange, zrange

def draw_triangles(fhash, tag, tris, imsize=(768, 1024)):
	canvas = np.zeros(imsize + (3,)).astype(np.uint8)
	for entry in tris:
		pts = entry['triangle'].reshape((-1,1,2)).astype(np.int32)
		cv2.fillPoly(canvas,[pts], [255, 255, 255])
	cv2.imwrite('_outputs/%s_%s.png' % (fhash, tag), canvas)