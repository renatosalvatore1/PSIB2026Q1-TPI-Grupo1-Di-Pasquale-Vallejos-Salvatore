#importo imagen desde url
import cv2
import urllib.request
import numpy as np

url = "https://www.pictoeduca.com/uploads/2017/07/eritrocitos.jpg" #erictrocitos
req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
image = cv2.imdecode(arr, -1)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


