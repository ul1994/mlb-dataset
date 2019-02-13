
from glob import glob
import cv2
import math, os, sys
import numpy as np
from configs import *
from utils import *
from scipy.ndimage import gaussian_filter as blur
import json

class Triangles:
	def __init__(self):
		pass

def find_joints(fhash, markupfiles=None, tag=None, spec=JOINTS_SPEC, imsize=IMAGE_SIZE, save=False):
	if tag is None: tag = fhash
	if markupfiles is None:
		markupfiles = sorted(glob('_meshes/*.obj'))
	filtered = frame_meshes(fhash, markupfiles)

	found_keypoints = []
	kplookup = {}
	all_triangles = {}
	person_triangles = []

	# mobj = mergedstruct()

	for file_ii, fname in enumerate(filtered):
		if primCount(fname) in BLACKLIST_MESHES:
			continue
		verts, faces, regs = load_markup(fname)

		objname = fname.replace('frame', 'blendf').replace('meshes', 'procd')
		notPerson = primCount(fname) not in active_meshes
		xrange, yrange, zrange = estSize(verts)
		tooBig = max(xrange, yrange, zrange) > maxEstSize

		for fii, face in enumerate(faces):
			if degenerative(face):
				continue
			triangle = np.array([verts[vi-1]['project'][:2] for vi in face])
			triangle[:, 1] = cpos(imsize[0], -triangle[:, 1])
			triangle[:, 0] = cpos(imsize[1], triangle[:, 0])
			zdepth = max([verts[vi-1]['project'][2] for vi in face])
			tri_id = '%s-%d' % (fname, fii)
			all_triangles[tri_id] = dict(
				fname=fname,
				zdepth=zdepth,
				triangle=triangle,
				face=fii,
			)

			# do not test for joints
			# merge_add(mobj, verts, faces)
			if notPerson or tooBig:
				continue

			person_triangles.append(tri_id)
			for kii, kp in enumerate(spec):
				for mcount, mrange in kp['match']:
					# print(mrange)
					if type(mrange) != list:
						mrange = list(mrange)
					if mcount == primCount(fname) and mrange[0] == fii:

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
								pos=loc,
								matched=kp,
								file=fname,
								pcount=primCount(fname)))

		sys.stdout.write('%s - %d/%d   \r' % (tag, file_ii, len(filtered)))
		sys.stdout.flush()

	result = Triangles()
	result.lookup = all_triangles
	result.all = list(all_triangles.values())
	result.people = [all_triangles[tid] for tid in person_triangles]
	return found_keypoints, result

def draw_joints(fhash, tag, kpoints, spec=JOINTS_SPEC, colored=True, save=False):
	import matplotlib.pyplot as plt

	bg = load_capture(fhash)
	bg = bg.astype(np.float32)
	imsize = bg.shape[:2]

	kpoints_overlay = np.zeros((3,) + imsize).astype(np.float32)
	text_layer = np.zeros(imsize + (3,), dtype=np.uint8)

	for kii, kp in enumerate(kpoints):
		joint_ind, (xx, yy, zz) = kp['joint_ind'], kp['pos']
		if yy >= imsize[0] or yy < 0 or xx >= imsize[1] or xx < 0:
			continue

		clr = CocoColors[joint_ind] if colored else [255, 255, 255]
		kpoints_overlay[:, int(yy), int(xx)] = np.array(clr)/255
		cv2.putText(text_layer, spec[joint_ind]['name'].lower(),
			(int(xx), int(yy)),
			font,
			fontScale,
			clr,
			lineType)


	kpoints_overlay = np.stack([blur(dim, 8) for dim in kpoints_overlay], 2)
	kpoints_overlay /= np.max(kpoints_overlay)
	# plt.figure(figsize=(14,14))
	# plt.imshow(kpoints_overlay)
	# plt.show(); plt.close()
	kpoints_overlay *= 255
	canvas = bg * 0.15 \
		+ kpoints_overlay * 0.6 \
		+ text_layer.astype(np.float32) * 0.25
	canvas = canvas.astype(np.uint8)
	if save:
		cv2.imwrite('_outputs/%s_%s.png' % (fhash, tag), bg)
	else:
		return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def draw_triangles(fhash, tris, tag='mask', imsize=IMAGE_SIZE, save=False):
	canvas = np.zeros(imsize + (3,)).astype(np.uint8)
	for entry in tris:
		pts = entry['triangle'].reshape((-1,1,2)).astype(np.int32)
		cv2.fillPoly(canvas,[pts], [255, 255, 255])
	if save:
		cv2.imwrite('_outputs/%s_%s.png' % (fhash, tag), canvas)
	else:
		return canvas

def collect_skeletons(raw_joints, skel_start = [326, 766]):
	skels = []
	bad = []
	track = []
	for ji, jnt in enumerate(raw_joints):
		if jnt['pcount'] in skel_start \
			and (ji == 0 or raw_joints[ji-1]['pcount'] not in skel_start):
			if len(track) == len(JOINTS_SPEC):
				skels.append(track)
			else:
				bad.append(track)
			track = []
		track.append(jnt)
	if len(track) == len(JOINTS_SPEC):
		skels.append(track)
	return skels, bad

def draw_skeletons(
	fhash, skels, centerfunc=None, imgcopy=False,
	outputDir='_outputs', captureDir='_captures', skel_size=14):
	global CocoPairsRender

	canvas = load_capture(fhash)
	imsize = canvas.shape
	# assert len(joints) % skel_size == 0

	for batch in skels:
		# batch = joints[sii:sii+skel_size]
		for entry in batch:
			joint_ind, (xx, yy, zdepth) = entry['joint_ind'], entry['pos']
			cv2.circle(
					canvas, (int(xx), int(yy)), 3, CocoColors[joint_ind],
					thickness=3, lineType=8, shift=0)

		skeleton = {}
		for entry in batch:
			skeleton[entry['matched']['name']] = entry
		quant = lambda name: (int(skeleton[name]['pos'][0]), int(skeleton[name]['pos'][1]))
		for pair_order, pair in enumerate(CocoPairsRender):
			disable_ankles = pair[0] not in [] and pair[1] not in []
			within_spec = pair[0] < len(JOINTS_SPEC) and pair[1] < len(JOINTS_SPEC)
			if within_spec and disable_ankles:
				cv2.line(
					canvas,
					quant(JOINTS_SPEC[pair[0]]['name']),
					quant(JOINTS_SPEC[pair[1]]['name']),
					CocoColors[pair_order], 3)
			# if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
			# 	continue

			# npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)

	# cv2.imwrite('%s/%s_skeleton.jpg' % (outputDir, fhash), canvas)
	return canvas

def draw_boxes(fhash, boxes, canvas):
	out = canvas.copy()
	for (x0, y0, xf, yf) in boxes:
		cv2.rectangle(out, (int(x0), int(y0)), (int(xf), int(yf)), [0, 0, 255], 3)
	return out

def draw_depth(fhash, tris, tag=None, imsize=IMAGE_SIZE, show=False):
	import matplotlib.pyplot as plt
	inds, zmin, zmax = depth_order(tris)

	canvas = np.zeros(imsize).astype(np.uint8)
	for ii in reversed(inds):
		entry = tris[ii]
		zd = entry['zdepth']
		zd -= zmin
		zd /= (zmax - zmin)
		zd = zd**0.5
		pts = entry['triangle'].reshape((-1,1,2)).astype(np.int32)

		cr = (1-zd) * 0.75 + 0.25
		cv2.fillPoly(canvas,[pts], 255 * cr)

	if show:
		plt.figure(figsize=(8, 8))
		plt.imshow(canvas)
		plt.show(); plt.close()
	return canvas

def save_metadata(fhash, skels, bounds, mask, droot='_data'):
	save_format = []
	for pi, person in enumerate(skels):
		obj = {}
		obj['joints'] = person
		for ji, joint in enumerate(obj['joints']):
			if 'matched' in joint:
				joint['name'] = joint['matched']['name']
				del joint['matched']
		obj['bounds'] = bounds[pi]
		save_format.append(obj)
	with open('%s/%s_joints.json' % (droot, fhash), 'w') as fl:
		json.dump(save_format, fl, indent=4)
	# with open('%s/%s_bounds.json' % (droot, fhash), 'w') as fl:
	# 	json.dump(bounds, fl, indent=4)
	np.save('%s/%s_mask.npy' % (droot, fhash), mask.astype(bool))
