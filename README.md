
# Deep Learning Final Project 1

### Gans-For-Image-Generation


Final Project for ECE-GY 7123 under the Guidance of Dr. Siddharth Garg and Dr. Arslan Mosenia in New York University.
* Mohammad Shamoon (ms12736@nyu.edu)
* Aradhya Alamuru (aa9405@nyu.edu)
* Umesh Deshmukh (urd7172@nyu.edu)


#### Install dependencies:--
```
pip install -r requirements.txt
```

## Run the Project
* To reproduce our trained model architecture, on MNIST
```python

model_path = './generator_state.pt'
trainedGenerator = Generator().cuda()
trainedGenerator.load_state_dict(torch.load(model_path, map_location=device), strict=False)
```

* To reproduce our trained model architecture, on pix2pix Map dataset
```python
python pix2pix.py
```

* To reproduce our trained model architecture, on Map dataset
```python
python eval.py
```

## pretrained weights
[pretrained weights](https://drive.google.com/drive/folders/1kEjxm4_6VhMk38Ux-6V9z7BmZU-YgoXu?usp=sharing)

## Dataset
[Dataset](https://drive.google.com/drive/folders/1yy1l0iwK5Y8Pc5pySpQy9nIzCKDfVvW_?usp=sharing)





