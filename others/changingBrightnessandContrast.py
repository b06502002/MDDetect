import cv2
### ref: https://www.geeksforgeeks.org/changing-the-contrast-and-brightness-of-an-image-using-python-opencv/

def BrightnessContrast(brightness=0):
	
	# getTrackbarPos returns the current
	# position of the specified trackbar.
	brightness = cv2.getTrackbarPos('Brightness',
									'GEEK')
	
	contrast = cv2.getTrackbarPos('Contrast',
								'GEEK')

	effect = controller(img, brightness,
						contrast)

	# The function imshow displays an image
	# in the specified window
	cv2.imshow('Effect', effect)

def controller(img, brightness=255,
			contrast=127):

	brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

	contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

	if brightness != 0:

		if brightness > 0:

			shadow = brightness

			max = 255

		else:

			shadow = 0
			max = 255 + brightness

		al_pha = (max - shadow) / 255
		ga_mma = shadow

		# The function addWeighted calculates
		# the weighted sum of two arrays
		cal = cv2.addWeighted(img, al_pha,
							img, 0, ga_mma)

	else:
		cal = img

	if contrast != 0:
		Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
		Gamma = 127 * (1 - Alpha)

		# The function addWeighted calculates
		# the weighted sum of two arrays
		cal = cv2.addWeighted(cal, Alpha,
							cal, 0, Gamma)

	# putText renders the specified text string in the image.
	cv2.putText(cal, 'B:{}, C:{}'.format(brightness,contrast), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

	return cal

if __name__ == '__main__':
	# The function imread loads an image
	# from the specified file and returns it.
	original = cv2.imread("/home/cov/Desktop/PML/project1_Mura/gimp img/2L_128_sImage_Image4_070.jpg")

	# Making another copy of an image.
	img = original.copy()

	# The function namedWindow creates a
	# window that can be used as a placeholder
	# for images.
	cv2.namedWindow('GEEK')

	# The function imshow displays an
	# image in the specified window.
	cv2.imshow('GEEK', original)

	# createTrackbar(trackbarName,
	# windowName, value, count, onChange)
	# Brightness range -255 to 255
	cv2.createTrackbar('Brightness',
					'GEEK', 255, 2 * 255,
					BrightnessContrast) # 365
	
	# Contrast range -127 to 127
	cv2.createTrackbar('Contrast', 'GEEK',
					127, 2 * 127,
					BrightnessContrast) # 217

	
	BrightnessContrast(0)

# The function waitKey waits for
# a key event infinitely or for delay
# milliseconds, when it is positive.
cv2.waitKey(0)
