import os
import numpy as np
import pandas as pd
import cv2

def get_data(filepath):
    all_imgs = {}

    classes_count = {}

    class_mapping = {}
    found_bg= False
    datapath = '/home/asprohy/data/traffic/train_trfc'
    label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    class_mapping = {
        'parkingLot': 0,
        'intersection': 1,
        'keepRight': 2,
        'leftOrRight': 3,
        'busPassage': 4,
        'leftDriving': 5,
        'slow': 6,
        'motorVehicleStraightOrRight': 7,
        'attentionToPedestrians': 8,
        'aroundTheIsland': 9,
        'straightOrRight': 10,
        'noBus': 11,
        'noMotorcycle': 12,
        'noMotorVehicle': 13,
        'noNonmotorvehicle': 14,
        'noHonking': 15,
        'interchangeStraightOrTurning': 16,
        'speedLimited40': 17,
        'speedLimited30': 18,
        'Honking': 19,
    }

    Revclass_mapping = {
        0: 'other',
        1: 'parkingLot',
        2: 'intersection',
        3: 'keepRight',
        4: 'leftOrRight',
        5: 'busPassage',
        6: 'leftDriving',
        7: 'slow',
        8: 'motorVehicleStraightOrRight',
        9: 'attentionToPedestrians',
        10: 'aroundTheIsland',
        11: 'straightOrRight',
        12: 'noBus',
        13: 'noMotorcycle',
        14: 'noMotorVehicle',
        15: 'noNonmotorvehicle',
        16: 'noHonking',
        17: 'interchangeStraightOrTurning',
        18: 'speedLimited40',
        19: 'speedLimited30',
        20: 'Honking',
    }

    print('startReadFile.')
    datas = pd.read_csv(label_path).values
    count = 0
    for c in datas[:1000]:
        filename, x1, y1, _, _, x2, y2, _, _, class_name = c

        class_name = Revclass_mapping[int(class_name)]
        filename = os.path.join(datapath,filename)

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if filename not in all_imgs:
            all_imgs[filename] = {}

            img = cv2.imread(os.path.join(filepath,filename))
            (rows, cols) = img.shape[:2]
            all_imgs[filename]['filepath'] = filename
            all_imgs[filename]['width'] = cols
            all_imgs[filename]['height'] = rows
            all_imgs[filename]['bboxes'] = []
            if np.random.randint(0, 6) > 0:
                all_imgs[filename]['imageset'] = 'trainval'
            else:
                all_imgs[filename]['imageset'] = 'test'
            #分一部分做测试

        all_imgs[filename]['bboxes'].append(
            {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        count += 1
        if count%100 ==0:
            print('count:',count)

    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    # make sure the bg class is last in the list
    if found_bg:
        if class_mapping['bg'] != len(class_mapping) - 1:
            key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
            val_to_switch = class_mapping['bg']
            class_mapping['bg'] = len(class_mapping) - 1
            class_mapping[key_to_switch] = val_to_switch

    # if 'bg' not in classes_count:
    #     classes_count['bg'] = 0
    #     class_mapping['bg'] = len(class_mapping)

    print('endReadFile.')


    return all_data, classes_count, class_mapping

def get_data2(input_path):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    visualise = False

    data_paths = [os.path.join(input_path,s) for s in ['VOC2012']]#'VOC2007',

    datapath = '/home/asprohy/data/traffic/train_trfc'
    label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    datas = pd.read_csv(label_path).values
    print('Parsing annotation files')

    for data_path in data_paths:

        annot_path = os.path.join(data_path, 'Annotations')
        imgs_path = os.path.join(data_path, 'JPEGImages')
        imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
        imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

        trainval_files = datas[:,0]
        test_files = []
        try:
            with open(imgsets_path_trainval) as f:
                for line in f:
                    trainval_files.append(line.strip() + '.jpg')
        except Exception as e:
            print(e)

        try:
            with open(imgsets_path_test) as f:
                for line in f:
                    test_files.append(line.strip() + '.jpg')
        except Exception as e:
            if data_path[-7:] == 'VOC2012':
                # this is expected, most pascal voc distibutions dont have the test.txt file
                pass
            else:
                print(e)

        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        idx = 0
        for annot in annots:
            try:
                idx += 1

                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                all_imgs.append(annotation_data)

                if visualise:
                    img = cv2.imread(annotation_data['filepath'])
                    for bbox in annotation_data['bboxes']:
                        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
                                                                          'x2'], bbox['y2']), (0, 0, 255))
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            except Exception as e:
                print(e)
                continue
    return all_imgs, classes_count, class_mapping
