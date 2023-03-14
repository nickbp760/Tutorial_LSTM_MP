import os

# specify the directory path
path = "./FaceVideoDataset"

# use os.listdir() to get a list of all files and folders in the directory
filesAndFolders = os.listdir(path)

# use a loop to iterate over the list and print out only the folders
for item in filesAndFolders:
    print(item)
    fullPath = os.path.join(path, item)  # get the full path of the item
    # if os.path.isdir(fullPath):  # check if the item is a directory
    #     print(item)
    Personfolders = os.listdir(fullPath)
    for personVideo in Personfolders:
        print("     ", personVideo)
        fullPathVideoList = os.path.join(fullPath, personVideo)  # get the full path of the item
        VideoList = os.listdir(fullPathVideoList)
        for videoFileName in VideoList:
            print("             ", videoFileName)
            fullPathVideoFileName = os.path.join(fullPathVideoList, videoFileName)  # get the full path of the item
