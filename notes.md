# overlay = np.zeros(imsize + (3,))
	# text_layer = np.zeros(imsize + (3,), dtype=np.uint8)

	# collision_inds = []
	# collision_triangles = []
	# for kii, kp in enumerate(found_keypoints):
	# 	joint_ind, (xx, yy, zz) = kp['joint_ind'], kp['pos']
	# 	if yy >= imsize[0] or yy < 0 or xx >= imsize[1] or xx < 0:
	# 		continue

	# 	occluded = False
	# 	for tii, (_, (tri, zdepth, fname)) in enumerate(all_triangles.items()):
	# 		pts = tri.reshape((-1,1,2)).astype(np.int32)
	# 		tri = [ls for ls in tri]
	# 		if zz < zdepth and inside_triangle(np.array([xx, yy]), *tri):
	# 			if tii not in collision_inds:
	# 				collision_inds.append(tii)
	# 				collision_triangles.append((pts, zdepth, fname))
	# 			occluded = True
	# 			break

	# 	if occluded:
	# 		overlay[int(yy), int(xx), 0] = 255
	# 	# else:
	# 		# overlay[int(yy), int(xx), :] = 255

	# 	cv2.putText(text_layer, kpinfo[joint_ind]['name'].lower(),
	# 		(int(xx), int(yy)),
	# 		font,
	# 		fontScale,
	# 		fontColor,
	# 		lineType)

	# 	sys.stdout.write('[%d/%d]   \r' % (kii, len(found_keypoints)))
	# sys.stdout.flush()
	# overlay = blur(overlay, 3)
	# overlay /= np.max(overlay)

	# styled_joints(fhash, 'collisions', found_keypoints)
	# bg = cv2.imread()
	# bg = bg[30:-10]
	# collision = np.zeros(imsize).astype(np.uint8)
	# collision = np.zeros(imsize).astype(np.uint8)
	# # print('Collisions: %d/%d     ' % (len(collision_inds), len(kplookup)))
	# for (pts, zdepth, fname) in collision_triangles:
	# 	print(zdepth, fname)
	# 	cv2.fillPoly(collision, [pts], 255)

	# bg += overlay * 255 * 0.75
	# cv2.imwrite('_outputs/%s_overlay.png' % fhash, bg)



	# cv2.imwrite('_outputs/%s_collision.png' % fhash, collision)