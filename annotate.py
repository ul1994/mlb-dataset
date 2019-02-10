
from glob import glob
import cv2
import math, os, sys
import numpy as np
from configs import *
from utils import *

class Triangles:
	def __init__(self):
		pass

def find_joints(tag, fhash, markupfiles, spec=JOINTS_SPEC, imsize=(768, 1024)):
	filtered = frame_meshes(fhash, markupfiles)

	found_keypoints = []
	kplookup = {}
	all_triangles = {}
	person_triangles = []

	# mobj = mergedstruct()

	for file_ii, fname in enumerate(filtered):
		verts, faces, regs = load_markup(fname)

		objname = fname.replace('frame', 'blendf').replace('meshes', 'procd')
		notPerson = pcount(fname) not in active_meshes
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
								pos=loc,
								matched=kp))

		sys.stdout.write('%s - %d/%d   \r' % (tag, file_ii, len(filtered)))
		sys.stdout.flush()

	result = Triangles()
	result.lookup = all_triangles
	result.all = list(all_triangles.values())
	result.people = [all_triangles[tid] for tid in person_triangles]
	return found_keypoints, result

def draw_skeletons(
	fhash, joints, centerfunc=None, imgcopy=False,
	outputDir='_outputs', captureDir='_captures'):
	global CocoPairsRender

	canvas = cv2.imread(captureDir + '/%s.jpg' % fhash)[30:-10]
	imsize = canvas.shape
	assert len(joints) % 12 == 0

	for sii in range(0, len(joints), 12):
		batch = joints[sii:sii+12]
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
			disable_ankles = pair[0] not in [10, 13] and pair[1] not in [10, 13]
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