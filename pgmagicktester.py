import pgmagick as pgm
import os

imagePath = '/home/cov/Desktop/PML/project1_Mura/AUO_Data/2nd/0826_2nd/DWL48/C53R5YA/C53R5YA_C3_3_PM_HGO_FMura.tif'
saveDir = '/home/cov/Desktop/PML/project1_Mura/AUO_Data/2nd/0826_2nd/DWL48/C53R5YA'

img = pgm.Image(imagePath)
img.depth(12) #sets to 10 bit

save_path = os.path.join(saveDir,'.'.join(["filenam3",'dng']))
img.write(save_path)