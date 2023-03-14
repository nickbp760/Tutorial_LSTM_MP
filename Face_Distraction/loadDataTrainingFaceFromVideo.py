import os

# specify the directory path
path = "./FaceVideoDataset"

# use os.listdir() to get a list of all files and folders in the directory
files_and_folders = os.listdir(path)

# use a loop to iterate over the list and print out only the folders
for item in files_and_folders:
    print(item)
    fullPath = os.path.join(path, item)  # get the full path of the item
    # if os.path.isdir(fullPath):  # check if the item is a directory
    #     print(item)
    Personfolders = os.listdir(fullPath)
    for personVideo in Personfolders:
        print("     ", personVideo)
        fullPathVideo = os.path.join(fullPath, item)  # get the full path of the item
