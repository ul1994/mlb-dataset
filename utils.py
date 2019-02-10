
from glob import glob
import numpy as np
from configs import *
import cv2
import math, os, sys

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

def filter_meshes(fhash, ls):
	assert len(ls) > 0
	filtered = []
	for ent in ls:
		fmatch = ent.split(sep)[-1].split('_')[0]
		# print(fmatch)
		if fhash == fmatch:
			filtered.append(ent)
#     print(filtered[0])
	# print(ent)
	# print(fmatch, fhash)
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

def dupeInds(face):
	ls = {}
	for vi in face:
		if vi not in ls:
			ls[vi] = 0
		ls[vi] += 1
	for _, count in ls.items():
		if count > 1:
			return True
	return False


def find_keypoints(tag, fhash, markupfiles, kpinfo, imsize=(768, 1024), captureDir = '_captures'):
	import matplotlib.pyplot as plt
	from scipy.ndimage import gaussian_filter as blur

	filtered = filter_meshes(fhash, markupfiles)

	pcount = lambda val: int(val.split('_')[-1].replace('.obj', ''))

	# found_keypoints = [list() for _ in kpinfo]
	found_keypoints = []

	mobj = mergedstruct()

	all_triangles = {}
	kplookup = {}
	triangles = {}
	for file_ii, fname in enumerate(filtered):
		verts, faces, regs = load_markup(fname)

		for fii, face in enumerate(faces):
			if dupeInds(face):
				continue
			triangle = np.array([verts[vi-1]['project'][:2] for vi in face])
			triangle[:, 1] = cpos(imsize[0], -triangle[:, 1])
			triangle[:, 0] = cpos(imsize[1], triangle[:, 0])
			zdepth = max([verts[vi-1]['project'][2] for vi in face])
			all_triangles['%d-%d' % (file_ii, fii)] = (triangle, zdepth)

		if pcount(fname) not in active_meshes:
			continue

		objname = fname.replace('frame', 'blendf').replace('meshes', 'procd')


#         vrange = 10000 * np.ones((3, 2))
#         vrange[:, 1] = -vrange[:, 1]
		wcoords = np.array([vert['world'] for vert in verts])
		xrange = np.max(wcoords[:, 0]) - np.min(wcoords[:, 0])
		yrange = np.max(wcoords[:, 1]) - np.min(wcoords[:, 1])
		zrange = np.max(wcoords[:, 2]) - np.min(wcoords[:, 2])
		estSize = max(xrange, yrange, zrange)
#         print(estSize)
		if estSize > maxEstSize:
			continue

		merge_add(mobj, verts, faces)


		for fii, face in enumerate(faces):
			triangle = np.array([verts[vi-1]['project'][:2] for vi in face])
			triangle[:, 1] = cpos(imsize[0], -triangle[:, 1])
			triangle[:, 0] = cpos(imsize[1], triangle[:, 0])
			zdim = [verts[vi-1]['project'][2] for vi in face]
			triangles['%d-%d' % (file_ii, fii)] = (triangle.copy(), zdim)
			# pts = triangle.reshape((-1,1,2)).astype(np.int32)

			for kii, kp in enumerate(kpinfo):
				for mcount, mrange in kp['match']:
					if type(mrange) == range:
						mrange = list(mrange)
					if mcount == pcount(fname) and mrange[0] == fii:

						flist = [face for _ii, face in enumerate(faces) if _ii in mrange]
						indlist = {}
						for tri in flist:
							for ind in tri:
								indlist[ind] = True
						zdim = []
						coords = np.zeros((2, len(indlist)))
						for ii, ind in enumerate(indlist.keys()):
							coords[:, ii] = verts[ind-1]['project'][:2]
							zdim.append(verts[ind-1]['project'][2])
						pnt = np.mean(coords, axis=1)
						zdepth = max(zdim)
						try:
							assert not math.isnan(pnt[0])
							assert not math.isnan(pnt[1])
						except:
							print(kp)
							print((mcount, mrange))
							print(pnt)
							assert False

						loc = (cpos(imsize[1], pnt[0]), cpos(imsize[0], -pnt[1]), zdepth)
						kpkey = '%.5f-%.5f' % (loc[0], loc[1])
						if kpkey not in kplookup:
							kplookup[kpkey] = 1
							found_keypoints.append(dict(
								joint_ind=kii,
								pos=loc))

		sys.stdout.write('%s - %d/%d   \r' % (tag, file_ii, len(filtered)))
		sys.stdout.flush()

	# FIXME: don't draw degenerative triangles
	canvas = np.zeros(imsize + (3,)).astype(np.uint8)
	for tid, (tri, zdim) in triangles.items():
		pts = tri.reshape((-1,1,2)).astype(np.int32)
		cv2.fillPoly(canvas,[pts], [255, 255, 255])
	cv2.imwrite('_outputs/%s_mask.png' % fhash, canvas)

	overlay = np.zeros(imsize + (3,))
	text_layer = np.zeros(imsize + (3,), dtype=np.uint8)

	collision_inds = []
	collision_triangles = []
	for kii, kp in enumerate(found_keypoints):
		joint_ind, (xx, yy, zz) = kp['joint_ind'], kp['pos']
		if yy >= imsize[0] or yy < 0 or xx >= imsize[1] or xx < 0:
			continue

		occluded = False
		for tii, (_, (tri, zdepth)) in enumerate(all_triangles.items()):
			pts = tri.reshape((-1,1,2)).astype(np.int32)
			tri = [ls for ls in tri]
			if zz < zdepth and inside_triangle(np.array([xx, yy]), *tri):
				if tii not in collision_inds:
					collision_inds.append(tii)
					collision_triangles.append(pts)
				occluded = True
				break

		if occluded:
			overlay[int(yy), int(xx), 0] = 255
		# else:
			# overlay[int(yy), int(xx), :] = 255

		cv2.putText(text_layer, kpinfo[joint_ind]['name'].lower(),
			(int(xx), int(yy)),
			font,
			fontScale,
			fontColor,
			lineType)

		sys.stdout.write('[%d/%d]   \r' % (kii, len(found_keypoints)))
	sys.stdout.flush()
	overlay = blur(overlay, 3)
	overlay /= np.max(overlay)

	collision = np.zeros(imsize).astype(np.uint8)
	# print('Collisions: %d/%d     ' % (len(collision_inds), len(kplookup)))
	for pts in collision_triangles:
		cv2.fillPoly(collision, [pts], 255)

	bg = cv2.imread(captureDir + '/%s.jpg' % fhash)
	bg = bg.astype(np.float32) * 0.25
	bg = bg[30:-10]
	bg[:, :, 2] += collision * 0.5 * 255
	bg += overlay * 255 * 0.75
	bg = bg * 0.75 + text_layer.astype(np.float32) * 0.25
	bg = bg.astype(np.uint8)
	cv2.imwrite('_outputs/%s_overlay.png' % fhash, bg)

	# cv2.imwrite('_outputs/%s_collision.png' % fhash, collision)

	return found_keypoints, mobj, triangles

def find_joint(jind, ls):
	for entry in ls:
		if entry['joint_ind'] == jind:
			return entry
	return None

def draw_skeletons(
	fhash, joints, centerfunc=None, imgcopy=False,
	outputDir='_outputs', captureDir='_captures'):
	global CocoPairsRender

	canvas = cv2.imread(captureDir + '/%s.jpg' % fhash)[30:-10]
	imsize = canvas.shape
	assert len(joints) % 12 == 0

	for entry in joints:
		joint_ind, (xx, yy, zdepth) = entry['joint_ind'], entry['pos']
		cv2.circle(
				canvas, (int(xx), int(yy)), 3, CocoColors[joint_ind],
				thickness=3, lineType=8, shift=0)

	# for ii in range(0, len(joints), 12):
	# 	human = joints[ii:ii+12]
	# 	for entry in human:
	# 		joint_ind, (xx, yy) = entry['joint_ind'], entry['pos']
	# 		if yy >= imsize[0] or yy < 0 or xx >= imsize[1] or xx < 0:
	# 			continue
	# 		cv2.circle(
	# 			canvas, (int(xx), int(yy)), 3, CocoColors[joint_ind],
	# 			thickness=3, lineType=8, shift=0)

		# # draw line
		# for pair_order, pair in enumerate(CocoPairsRender):
		# 	j0 = find_joint(pair[0], human)
		# 	j1 = find_joint(pair[1], human)
		# 	if j0 is None or j1 is None:
		# 		continue
		# 	j0 = j0['pos']
		# 	j1 = j1['pos']
		# 	if j0[1] >= imsize[0] or j0[1] < 0 or j0[0] >= imsize[1] or j0[0] < 0:
		# 		continue
		# 	if j1[1] >= imsize[0] or j1[1] < 0 or j1[0] >= imsize[1] or j1[0] < 0:
		# 		continue

		# 	cv2.line(
		# 		canvas,
		# 		(int(j0[0]), int(j0[1])),
		# 		(int(j1[0]), int(j1[1])),
		# 		CocoColors[pair_order], 3)

	cv2.imwrite('%s/%s_skeleton.jpg' % (outputDir, fhash), canvas)
	return canvas
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

# def merge_all(merged, fname, prop):
#     with open(fname, 'w') as fl:
#         print('Verts', len(merged.verts))
#         for vert in merged.verts:
#             fl.write('v %s\n' % ' '.join(['%f' % val for val in vert[prop][:3]]))
#         for face in merged.faces:
#             fl.write('f %s\n' % ' '.join(['%d' % val for val in face]))

# markupfiles = sorted(glob(meshDir + '/frame*.obj'))
# framefiles = sorted(glob(captureDir + '/*.jpg'))

# for fname in framefiles:
#     fcount = int(fname.split('\\')[-1][:-4])
#     merged = merge_frame(fcount)
#     merge_all(merged, 'merged.obj', 'world')
#     break