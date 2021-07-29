import os
import cv2
import face_recognition
import numpy as np

########### INIT ################
path_faces = "ParickovaFaces"
# path_faces = "HBS_Faces"
path_fotos = "ParickovaFotos"
#path_fotos = "HBS_Fotos"
path_rating = "ParickovaRating"
#path_rating = "HBS_Rating"
######## FUNCTIONS ##############

def makeParickovaLists(path):
    parickovaFaces = []
    parickovaNames = []
    myList = os.listdir(path)
    for face in myList:
        curImg = cv2.imread(f"{path}/{face}")
        parickovaFaces.append(curImg)
        parickovaNames.append(str.split(face, sep=".")[0])
    return parickovaFaces, parickovaNames


def findParickovaEncodings(parickovaFaces):
    encodeList = []
    for img in parickovaFaces:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList


def makeFotoList(path):
    fotoFaceLoc = []
    fotoFaceEncode = []
    fotoNames = []
    fotoList = os.listdir(path)
    for foto in fotoList:
        fotoNames.append(foto)
        foto = face_recognition.load_image_file(f"{path}/{foto}")
        foto = cv2.cvtColor(foto, cv2.COLOR_BGR2RGB)
        faceLoc = face_recognition.face_locations(foto)
        encodeFoto = face_recognition.face_encodings(foto, faceLoc)
        fotoFaceLoc.append(faceLoc)
        fotoFaceEncode.append(encodeFoto)
    return fotoNames, fotoFaceLoc, fotoFaceEncode


############## CORE CALL ################
# search for faces, names and encodings of parickova members
parickovaFaces, parickovaNames = makeParickovaLists(path_faces)
parickovaEncode = findParickovaEncodings(parickovaFaces)

# search of location and encodings of faces in the pictures [picName.][pictureNr.][faceNr.]
parickovaFotosNames, parickovaFotosLoc, parickovaFotosEncode  = makeFotoList(path_fotos)

# searching, whether parickova members are on pictures.
for curPic, curEnc, curLoc in zip(parickovaFotosNames, parickovaFotosEncode, parickovaFotosLoc):
    img = cv2.imread(f"{path_fotos}/{curPic}")
    print("In", curPic, "are", len(curEnc), "faces")
    for i, loc in zip(curEnc, curLoc):
        sourceFoto = ""
        nameFolder = ""
        targetFoto = ""
        compareResult = face_recognition.compare_faces(parickovaEncode, i)
        faceDistance = face_recognition.face_distance(parickovaEncode, i)
        minIdx = np.argmin(faceDistance)
        if compareResult[minIdx] == True:
            print("There is:", parickovaNames[minIdx])
            cv2.putText(img, parickovaNames[minIdx], (loc[3], loc[2]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            sourceFoto = os.path.join(path_fotos, curPic)
            nameFolder = (f"{path_rating}/{parickovaNames[minIdx]}")
            targetFoto = (f"{nameFolder}/{str.split(curPic, sep='.')[0]}_{parickovaNames[minIdx]}.jpg")
            print(f"source pic: {sourceFoto} dest pic: {targetFoto}")
            if os.path.isdir(nameFolder) == True:
                print(f"Folder {nameFolder} exists. Creating symlink from {sourceFoto} to {targetFoto}")
                os.link(sourceFoto, targetFoto)
            else:
                print(f"Folder {nameFolder} doesnt exists. Creating folder.")
                os.makedirs(nameFolder)
                print(f"Folder {nameFolder} created. Creating symlink.")
                os.link(sourceFoto, targetFoto)
        else:
            print("I dont know who is at", loc)
            cv2.rectangle(img, (loc[3], loc[0]), (loc[1], loc[2]), (0, 255, 0), 2)

    cv2.imwrite(f"{path_rating}/{str.split(curPic, sep='.')[0]}_rated.jpg", img)

