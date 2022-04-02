


maskDetector = load_model("model_mask.model")

path_proto = r"C:\Users\HAZAL\Desktop\facemask_detection-main\protomodels\deploy.prototxt"
path_coffe = r"C:\Users\HAZAL\Desktop\facemask_detection-main\protomodels\res10_300x300_ssd_iter_140000.caffemodel"
faceDetect = cv2.dnn.readNet(path_proto, path_coffe)


print("\n")
print("\n")
print(" Face mask detector will start in 10 seconds...")
print("\n")
print(" If you want to close face mask detector, PRESS 'x' while the frame window is open... ")
print("\n")
print("\n")

vs = VideoStream(src=0).start()
time.sleep(10)



def maskDetect(frameDim, faceDetect, maskDetector):

	(h, w) = frameDim.shape[:2]
	blob = cv2.dnn.blobFromImage(frameDim, 1.0, (200, 200),
								 (104.0, 177.0, 123.0))

	faceDetect.setInput(blob)
	detections = faceDetect.forward()
	print(detections.shape)

	faces = []
	locations = []
	predicts = []

	for i in range(0, detections.shape[2]):

		preview = detections[0, 0, i, 2]

		if preview > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frameDim[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (200, 200))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locations.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		predicts = maskDetector.predict(faces, batch_size=32)
	return (locations, predicts)


while True:
	frameDim = vs.read()
	frameDim = imutils.resize(frameDim, width=800)



	(locations, predicts) = maskDetect(frameDim, faceDetect, maskDetector)

	for (box, pred) in zip(locations, predicts):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred


		if mask > withoutMask:
			label = "Wearing Mask"
			color = (255, 0, 0)
		else:
			label = "Not Wearing Mask"
			color = (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frameDim, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frameDim, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Face Mask Detector", frameDim)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("x"):
		break


cv2.destroyAllWindows()
vs.stop()