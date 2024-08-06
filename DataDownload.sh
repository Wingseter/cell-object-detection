apt-get upgrade && apt-get update
pip install kaggle
mkdir Data
kaggle competitions download -c sartorius-cell-instance-segmentation
apt-get install unzip -y
unzip sartorius-cell-instance-segmentation.zip -d Data
rm -f sartorius-cell-instance-segmentation.zip