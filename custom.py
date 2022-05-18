
def createDataset(drivePath, savePath, lastImage):
      for i in range(1, lastImage):
        name = str(i) + '.jpg'
        imgName = drivePath + name
        print(imgName)
        imgPath = os.path.join(drivePath, imgName)
        imgArr = np.array(Image.open(imgPath))
        target = imgArr[:, 600:, :]
        imageio.imwrite(savePath + str(i) + '.png', target)

path = '/content/drive/MyDrive/pix2pix/train/'
path_new = '/content/drive/MyDrive/pix2pix/new/'
lastImage = 1097;
createDataset(
    path, 
    path_new,
    lastImage
)