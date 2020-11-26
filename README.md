# Beer Consumption

Notebook and Streamlit application used as small demonstration for Transformation Academy course.

The dataset used in this experiment was release originally at [Kaggle](https://www.kaggle.com/dongeorge/beer-consumption-sao-paulo).


## Getting Started

Both applications has the same goal: test a small regression demonstration and allow some explorations using Notebook or Streamlit interactions.
In order to open the Notebook make sure to have the environmental requirements, you can use Anaconda. For Streamlit make sure to have the same requirements including the Streamlit libraries.

All requirements are listed in ```requirements.txt``` 

For this repository we also deploy in Streamlit cloud services, so you can access directly using this [link](https://share.streamlit.io/eylatamsouth/ta-beer-consuption).


### Screenshots

Streamlit has several UI components to allow interaction.

![](img/demo.gif)

### Prerequisites

Install a Python > =3.6 environment (in case you don't have one).

```
python3 -m venv .venv
source .venv/bin/activate
```

Then install the requirements.

```
pip install requirements.txt
```

### Installing

For Streamlit local usage, first install the library:

```
pip install streamlit
```

Then, use the ```streamlit``` command to start the application of the file ```streamlit_app.py```:

```
streamlit streamlit_app.py
```

After that will be create a local server to allow interactions using the web browser.

## Deployment

Changes in the master branch will trigger a new deployment in Streamlit pipeline.

## Built With

* [Python 3.6+](https://python.org) and virtual environment
* [Streamlit](https://www.streamlit.io/) - Sharing data and interaction
* [Kaggle](kaggle.com/) - Dataset

## Contributing

Please feel free to propose new features by raising an [Issue](https://github.com/EYLatamSouth/ta-beer-consuption/issues/new/choose) or creating a Pull Request.

## Authors

* **Michel Fernandes** - *Initial work* - [michelpf](https://github.com/michelpf)

See also the list of [contributors](https://github.com/EYLatamSouth/ta-beer-consuption/contributors) who participated in this project.