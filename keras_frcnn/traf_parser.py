import os
import numpy as np
import pandas as pd
import cv2

def get_data(filepath):
    all_imgs = {}

    classes_count = {}

    class_mapping = {}
    found_bg= False

    label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    class_mapping = {
        'other': 0,
        'parkingLot': 1,
        'intersection': 2,
        'keepRight': 3,
        'leftOrRight': 4,
        'busPassage': 5,
        'leftDriving': 6,
        'slow': 7,
        'motorVehicleStraightOrRight': 8,
        'attentionToPedestrians': 9,
        'aroundTheIsland': 10,
        'straightOrRight': 11,
        'noBus': 12,
        'noMotorcycle': 13,
        'noMotorVehicle': 14,
        'noNonmotorvehicle': 15,
        'noHonking': 16,
        'interchangeStraightOrTurning': 17,
        'speedLimited40': 18,
        'speedLimited30': 19,
        'Honking': 20,
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

    print('endReadFile.')

    return all_data, classes_count, class_mapping