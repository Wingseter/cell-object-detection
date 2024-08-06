pip install kaggle
mkdir Data
kaggle competitions download -c sartorius-cell-instance-segmentation
unzip sartorius-cell-instance-segmentation.zip -d Data
rm -f sartorius-cell-instance-segmentation.zip