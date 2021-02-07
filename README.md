# NLP-Microservice
An example for a Deep Learning NLP model hosted as a RESTful API.   

## Python Environment
- Conda environment: `conda create -n nlp-ms python=3.8`
- Dependencies:
  - API Stuff:
    - [Flask](https://flask.palletsprojects.com/en/1.1.x/): `pip install Flask`
    - [Flask-RESTX](https://flask-restx.readthedocs.io/en/latest/): `pip install flask-restplus`
  - [🤗Transformers](https://github.com/huggingface/transformers) is used as NLP library: `pip install transformers`
  - [TensorFlow 2.0]() or [PyTorch]()
    - I used Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation instructions) 
  
## Tutorial for Creating such a Microservice
Please refer to the folder [tutorial](./tutorial).