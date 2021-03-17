import xml.etree.ElementTree as ET
import io
import os
import csv
import glob
import pandas as pd
import tensorflow as tf
import PIL.Image
import numpy as np
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from pandas import DataFrame
from collections import namedtuple

def splitt(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def xmltocsv(annotations_path, train_or_test_path, sub_folder_name, labels, labels_no):
    # tensorflowun anlayabileceği formata dönüşüm için
    file_path = train_or_test_path  # test ve train dosyalarının ortak klasörleri
    new_file = open('{0}.csv'.format(annotations_path + '/' + format(sub_folder_name)), mode='w',
                    newline='')  # train ya da test hangi kalasördeyse sıra, oraya o isimle csv file aç
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    rows = []  # xmllist
    for xml_file in glob.glob(file_path + '/*.xml'):  # o file pathteki bütün isimlere sahip dosyalar
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for h_command in root.findall('object'):
            v = (root.find('filename').text,
                 int(root.find('size')[0].text),
                 int(root.find('size')[1].text),
                 h_command[0].text,
                 #root.find('./object')[0].text,
                 int(root.find('./object')[4][0].text),
                 int(root.find('./object')[4][1].text),
                 int(root.find('./object')[4][2].text),
                 int(root.find('./object')[4][3].text)
                 )
            rows.append(v)
    csv_writer = csv.writer(new_file)
    # csv_writer.writerow(labels)  # csv başlık satırı    #writerow(sütunlar)
    # csv_writer.writerow(column_name)  # csv başlık satırı    #writerow(sütunlar)
    # for row in rows:
    #    csv_writer.writerow(row)

    # print(rows)

    xml_df = pd.DataFrame(rows,columns=column_name)
    print(xml_df)   #examples


    xml_df.to_csv('{0}.csv'.format(annotations_path + '/' + format(sub_folder_name)))

    output_path = os.path.join(annotations_path, '{0}.record'.format(sub_folder_name))  # .recordlar
    csv_path = os.path.join(annotations_path, '{0}.csv'.format(sub_folder_name))
    desk_path='C:/Users/ASUS/Desktop'
    input_path = os.path.join(desk_path, '{0}'.format(sub_folder_name))  # image_dir
    print(input_path)  # train ya da test klasörleri
    print(csv_path)  # train.csv ya da test.csv
    print(output_path)  # train.tfrecord ya da test.tfrecord

    #examples = pd.read_csv(csv_path)
    examples=xml_df
    print(examples)  # tablo ve 6 rows 10 columns

    # Writes a dataset to a TFRecord file.
    writer = tf.io.TFRecordWriter(
        output_path)  # <tensorflow.python.lib.io.tf_record.TFRecordWriter object at 0x0000020AE59FDD60>
    # writer = tf.io.gfile.GFile(tf.io.TFRecordWriter(path=output_path))
    # #<tensorflow.python.platform.gfile.GFile object at 0x000001DC145D8940>
    # tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(filename, 'utf-8')]))
    print('Writer:')
    print(writer)

    grouped = splitt(examples, 'filename')
    print(grouped)

    filenames = []
    for group in grouped:

        with tf.io.gfile.GFile(os.path.join(input_path, '{}'.format(group.filename)), 'rb') as fid:
            print(type(fid))
            print('fid')
            print(fid)
            jpg_image = fid.read()
        print('beyz')

        jpg_image_io = io.BytesIO(jpg_image)
        image = PIL.Image.open(jpg_image_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        else:
            print('yessssssssssss')
        width, height = image.size
        print(width, height)

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        labels_path="C:/Users/ASUS/RealTimeObjectDetection/Tensorflow/workspace/annotations/label_map.pbtxt"
        label_map = label_map_util.get_label_map_dict(labels_path)
        print(label_map)

        def class_text_to_int(row_label):
            if row_label == 'bir':
                return 1
            elif row_label == 'iki':
                return 2
            elif row_label == 'uc':
                return 3
            elif row_label == 'dort':
                return 4
            elif row_label == 'bes':
                return 5
            elif row_label == 'alti':
                return 6
            elif row_label == 'yedi':
                return 7
            elif row_label == 'sekiz':
                return 8
            elif row_label == 'dokuz':
                return 9
            elif row_label == 'sifir':
                return 10
            elif row_label == 'kanalartir':
                return 11
            elif row_label == 'kanalazalt':
                return 12
            elif row_label == 'sesac':
                return 13
            elif row_label == 'seskapa':
                return 14
            elif row_label == 'ackapa':
                return 15
            else:
                return None
        for index, row in group.object.iterrows():
            number=index
            print(type(classes))
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            print(row['class'])
            #classes_text.append(row['class'])
            #classes.append(label_map_dict[row['class']])
            #classes_text.append(row['class'])
            classes_text.append(str(row['class']).encode('utf8'))
            print('**********CLASS:')
            print(row['class'])
            print(class_text_to_int(row['class']))
            classes.append(class_text_to_int(row['class']))


        feature = {'image/height': dataset_util.int64_feature(height),
                   'image/width': dataset_util.int64_feature(width),
                   'image/filename': dataset_util.bytes_feature(filename),
                   'image/source_id': dataset_util.bytes_feature(filename),
                   'image/encoded': dataset_util.bytes_feature(jpg_image),
                   'image/format': dataset_util.bytes_feature(image_format),      #jpg
                   'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                   'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                   'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                   'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                   'image/object/class/text': dataset_util.bytes_list_feature(classes_text),   #cat,dog vs..
                   'image/object/class/label': dataset_util.int64_list_feature(classes)        #1,2 vs
                   }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        # return example_proto.SerializeToString()
        #print(example_proto.SerializeToString())
        print(feature)
        writer.write(example_proto.SerializeToString())
        print('******************************************')

    print('Successfull TFRecords: {}'.format(output_path))
    writer.close()