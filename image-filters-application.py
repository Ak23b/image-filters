import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading the image
img = cv2.imread("uamag0n4.png")

# Defining Kernels
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

sobel_y = np.array([[-1,-2,-2],
                    [0,0,0],
                    [1,2,1]])

prewitt_x = np.array([[-1,0,1],
                      [-1,0,1],
                      [-1,0,1]])

prewitt_y = np.array([[-1,-1,-1],
                      [0,0,0],
                      [1,1,1]])

laplacian = np.array([[0,-1,0],
                      [-1,4,-1],
                      [0,-1,0]])

# Apply filters
sobel_x_res = cv2.filter2D(img,-1,sobel_x)
sobel_y_res = cv2.filter2D(img,-1,sobel_y)
sobel_res = cv2.magnitude(sobel_x_res.astype(np.float32),sobel_y_res.astype(np.float32))


prewitt_x_res = cv2.filter2D(img,-1,prewitt_x)
prewitt_y_res = cv2.filter2D(img, -1, prewitt_y)
prewitt_res = cv2.magnitude(prewitt_x_res.astype(np.float32), prewitt_y_res.astype(np.float32))

laplacian_res = cv2.filter2D(img,-1, laplacian)


# Show the results
plt.figure(figsize=(12,8))
plt.subplot(2,2,1); plt.imshow(sobel_res,cmap='gray'); plt.title('Sobel')
plt.subplot(2,2,2); plt.imshow(prewitt_res,cmap='gray'); plt.title('Prewitt')
plt.subplot(2,2,3); plt.imshow(laplacian_res,cmap='gray'); plt.title("Laplacian")
plt.subplot(2,2,4); plt.imshow(img,cmap='gray'); plt.title("Original")
plt.show()