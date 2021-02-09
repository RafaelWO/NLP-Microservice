# NLP-Microservice
An example for a Deep Learning NLP model hosted as a RESTful API.   

## Python Environment
- Conda environment: `conda create -n nlp-ms python=3.8`
- Dependencies:
  - Run `pip install -r ./requirements.txt` to install all dependencies (except PyTorch)
  - API Stuff:
    - [Flask](https://flask.palletsprojects.com/en/1.1.x/)
    - [Flask-RESTX](https://flask-restx.readthedocs.io/en/latest/)
  - [🤗Transformers](https://github.com/huggingface/transformers) is used as NLP library
  - [TensorFlow 2.0]() or [PyTorch]()
    - I used Pytorch (see [here](https://pytorch.org/get-started/locally/) for installation instructions)
  
## Usage
To run the Text-Generation-API simply execute `python src/service.py` in the root directory.

The script `sample_request.py` can be used to issue requests to a running Text-Generation-API: 
`python sample_request.py <api-host-or-ip-adress>`.
  
## Tutorial for Creating such a Microservice
Please refer to the folder [tutorial](./tutorial).