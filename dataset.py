
from glob import glob
import cv2
import math, os, sys
import numpy as np
from configs import *
from utils import *
from scipy.ndimage import gaussian_filter as blur
import json
from keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa

class MLBDataset(Sequence):
	version = '1.0'

	def __init__(self, bsize,
		# 480 / 16 = 30
		# 256 / 16 = 16
		insize=(480, 256), outratio=16,
		maxPeople=5, # ignore captures with tons of ppl
		dpath='_data'):

		self.bsize = bsize
		self.insize = insize
		self.inhalf = insize[0]//2, insize[1]//2
		self.outratio = outratio
		self.captures = gather_captures()
		def aug_make():
			return iaa.Sequential([
				iaa.Affine(
					translate_percent=(-0.35, 0.35),
					rotate=(-15, 15),
					scale=(0.7, 0.8)),
			]).to_deterministic()
		self.aug = aug_make
		assert len(self.captures)
		print('MLB Dataset (Version %s)' % self.version)
		print(' [*] Found captures: %d' % len(self.captures))

		self.people = []
		for ci, capid in enumerate(self.captures):
			metapath = '%s/%s_joints.json' % (dpath, capid)
			with open(metapath) as fl:
				meta = json.load(fl)
			assert len(meta)
			if len(meta) > maxPeople:
				continue
			for ji in range(len(meta)):
				self.people.append((ci, ji, metapath))
		print(' [*] Contains people: %d' % len(self.people))

	def __len__(self):
		return len(self.people)

	def __getitem__(self, idx):
		refs = self.people[idx:idx + self.bsize]
		images = []
		masks = []
		points = []
		parity = []
		for capid, pid, mpath in refs:
			# TODO: determine aug
			with open(mpath) as fl:
				meta = json.load(fl)
			person = meta[pid]
			cap = self.captures[capid]
			x0, y0, xf, yf = person['bounds']
			# bh = yf - y0
			wh, hh = self.inhalf
			cx, cy = (midpnt(xf, x0), midpnt(yf, y0))

			img = load_capture(cap)
			cv2.rectangle(img, (int(x0), int(y0)), (int(xf), int(yf)), [0, 0, 255], 2)
			cv2.circle(img, (int(cx), int(cy)), 10, [0, 255, 255], 2)

			# ratio bounds
			rHeight = yf - y0
			hscale = self.insize[1] / rHeight
			rWidth = self.insize[0] / hscale
			# print(rWidth, rHeight)
			cutX = int(cx - rWidth/2), int(cx + rWidth/2)
			cutY = int(cy - rHeight/2), int(cy + rHeight/2)

			def fit_img(img, cutX, cutY, cdim=3, dtype=np.uint8):
				ypad, xpad = [max(-cutY[0], 0), max(-cutX[0], 0)] # y0, x0
				cutImg = img[max(cutY[0], 0):cutY[1], max(cutX[0], 0):cutX[1]]
				cutImg = cv2.resize(cutImg, (0,0), fx=hscale, fy=hscale)
				cutImg = cutImg.reshape(cutImg.shape[:2] + (cdim,))
				cutImg = cutImg[:self.insize[1], :self.insize[0]]
				canvas = np.zeros((self.insize[1], self.insize[0], cdim,)).astype(dtype)
				# print(rWidth, rHeight, xpad*hscale, ypad*hscale)
				canvas[
					int(ypad*hscale):int(ypad*hscale)+cutImg.shape[0],
					int(xpad*hscale):int(xpad*hscale)+cutImg.shape[1]] = cutImg
				return canvas

			recenter = lambda val, fromCent, toCent, scl: (val - fromCent) * scl + toCent

			augfunc = self.aug()
			cropIm = fit_img(img, cutX, cutY)
			mask = fit_img(np.ones(img.shape[:2] + (1,)), cutX, cutY, cdim=1)
			kpimage = np.zeros((self.insize[1], self.insize[0], len(JOINTS_SPEC)))
			pairImage = [np.zeros((self.insize[1], self.insize[0])) for _ in range(len(CocoPairs))]
			for person in meta:
				skeleton = {}
				pointByName = lambda name: (int(skeleton[name][0]), int(skeleton[name][1]))
				for joint in person['joints']:
					jii, (imX, imY, zd) = joint['joint_ind'], joint['pos']
					yy = recenter(imY, cy, self.insize[1]/2, hscale)
					xx = recenter(imX, cx, self.insize[0]/2, hscale)
					skeleton[JOINTS_SPEC[jii]['name']] = (xx, yy) # used later for pairs
					if yy >= 0 and yy < self.insize[1] and xx >= 0 and xx < self.insize[0]:
						place_blur(kpimage[...,jii], int(xx), int(yy))
				for pair_order, pair in enumerate(CocoPairs):
					cv2.line(
						pairImage[pair_order],
						pointByName(JOINTS_SPEC[pair[0]]['name']),
						pointByName(JOINTS_SPEC[pair[1]]['name']),
						255, 3)
			pairImage = np.transpose(pairImage, (1, 2, 0)).astype(np.float32)/255
			aug_result = augfunc.augment_image(
				np.concatenate([
					cropIm,
					mask,
					kpimage,
					pairImage,
				], axis=2))
			augIm = aug_result[...,:3]
			augMask = aug_result[...,3:4]
			augPoints = aug_result[...,4:4+len(JOINTS_SPEC)]
			augPairs = aug_result[...,4+len(JOINTS_SPEC):]

			images.append(augIm)
			masks.append(augMask)
			points.append(augPoints)
			parity.append(augPairs)

		images = np.array(images).astype(np.uint8)
		masks = np.array(masks).astype(np.uint8)
		points = np.array(points)
		# for ent in parity:
		# 	print(ent.shape)
		parity = np.array(parity)

		return (images, masks), (points, parity)

	def on_epoch_end(self):
		pass

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	dset = MLBDataset(1)

	images, labels = dset[0]

	# plt.figure(figsize=(8, 8))
	# plt.imshow(images[0])
	# plt.show()
	cv2.imwrite('dump.jpg', images[0])
