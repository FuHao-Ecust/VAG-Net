organ_ls = ['Liver','Right kidney','Left kidney','Spleen']
organ_dict = {'background':0,'Liver':63,'Right kidney':126,'Left kidney':189,'Spleen':252}

label2trainid = {0:0,63:1, 126:2, 189:3, 252:4}
train2label = {0:0,1:63,2:126,3:189,4:252}
num_classes = 5

MRI_ls = ['T1DUAL', 'T2SPIR']

DICOM_dir = './dicom'
LOG_dir = './logger' # save logger path