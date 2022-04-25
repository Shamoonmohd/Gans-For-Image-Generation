
# Deep Learning Final Project 1(Under Development)

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
* To reproduce our trained model architecture, on MNIST dataset
```python

model_path = './generator_state.pt'
trainedGenerator = Generator().cuda()
trainedGenerator.load_state_dict(torch.load(model_path, map_location=device), strict=False)
```



