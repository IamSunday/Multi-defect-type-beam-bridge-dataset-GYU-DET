from xml.dom.minidom import Document
import os
import cv2


def makexml(picPath, txtPath, xmlPath):
    """
    Convert YOLO-format .txt annotations to VOC-format .xml annotations.
    :param picPath: path to the images
    :param txtPath: path to the label .txt files
    :param xmlPath: path to save generated .xml files
    """
    # Mapping class indices to class names — must match the order in your classes.txt
    dic = {'0': "Crack",
           '1': "Breakage",
           '2': "fengwo",
           '3': "Comb",
           '4': "Hole",
           '5': "Reinforcement",
           '6': "Seepage"}

    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # Root tag
        xmlBuilder.appendChild(annotation)

        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        img = cv2.imread(picPath + name[0:-4] + ".jpg")
        Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)

        filename = xmlBuilder.createElement("filename")
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)

        size = xmlBuilder.createElement("size")
        width = xmlBuilder.createElement("width")
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)

        height = xmlBuilder.createElement("height")
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)

        depth = xmlBuilder.createElement("depth")
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)

        annotation.appendChild(size)

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")

            picname = xmlBuilder.createElement("name")
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)

            pose = xmlBuilder.createElement("pose")
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)

            truncated = xmlBuilder.createElement("truncated")
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)

            difficult = xmlBuilder.createElement("difficult")
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)

            bndbox = xmlBuilder.createElement("bndbox")

            xmin = xmlBuilder.createElement("xmin")
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)

            ymin = xmlBuilder.createElement("ymin")
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)

            xmax = xmlBuilder.createElement("xmax")
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)

            ymax = xmlBuilder.createElement("ymax")
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)

            object.appendChild(bndbox)
            annotation.appendChild(object)

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


if __name__ == "__main__":
    picPath = "F:/voc2014/JPEGImages/"  # Path to image files (include trailing slash)
    txtPath = "F:/voc2014/labels/"      # Path to YOLO-format label .txt files
    xmlPath = "F:/voc2014/Annotations/" # Path to save generated VOC-format .xml files
    makexml(picPath, txtPath, xmlPath)
