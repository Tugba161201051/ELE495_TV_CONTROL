import tf_record as xc
import os

annotations_path = 'C:/Users/ASUS/RealTimeObjectDetection/Tensorflow/workspace/annotations'
images_path = 'C:/Users/ASUS/RealTimeObjectDetection/Tensorflow/workspace/images'
photos_path = images_path + '/photos'
labels = os.listdir(photos_path)  #"komutlar" komutların arrayi için uzunluğuna bakıcağımız liste
labels_no = [i+1 for i in range (len(labels))]   #[1,2...]
labels_dictionary = {'id':1,'name':'bir'}, {'id':2,'name':'iki'}, {'id':3,'name':'uc'}, {'id':4,'name':'dort'},{'id':4,'name':'dort'},{'id':5,'name':'bes'}, {'id':6,'name':'alti'}, {'id':7,'name':'yedi'}, {'id':8,'name':'sekiz'}, {'id':9,'name':'dokuz'}, {'id':10,'name':'sifir'},{'id':11,'name':'kanalartir'}, {'id':12,'name':'kanalazalt'}, {'id':13,'name':'sesac'},{'id':14,'name':'seskapa'},{'id':15,'name':'ackapa'}
labels_properties =[{'id':1,'name':'bir'}, {'id':2,'name':'iki'}, {'id':3,'name':'uc'}, {'id':4,'name':'dort'},{'id':4,'name':'dort'},{'id':5,'name':'bes'}, {'id':6,'name':'alti'}, {'id':7,'name':'yedi'}, {'id':8,'name':'sekiz'}, {'id':9,'name':'dokuz'}, {'id':10,'name':'sifir'},{'id':11,'name':'kanalartir'}, {'id':12,'name':'kanalazalt'}, {'id':13,'name':'sesac'},{'id':14,'name':'seskapa'},{'id':15,'name':'ackapa'}]

doc = open("C:/Users/ASUS/RealTimeObjectDetection/Tensorflow/workspace/annotations/label_map.pbtxt", "w")
for label in labels_properties:
    doc.write('item { \n')
    doc.write('\tname:\'{}\'\n'.format(label['name']))
    doc.write('\tid:{}\n'.format(label['id']))
    doc.write('}\n')
doc.close()

sub_folder_name = 'train'
train_path='C:/Users/ASUS/Desktop/train'
xc.xmltocsv(annotations_path,train_path,sub_folder_name,labels,labels_no)

sub_folder_name = 'test'
test_path='C:/Users/ASUS/Desktop/test'
xc.xmltocsv(annotations_path,test_path,sub_folder_name,labels,labels_no)