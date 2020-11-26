# Predicting Flight Delays Using Weather Data and Machine Learning

##### Project source code for the NTNU course TDT4173 - Machine Learning

The project was conducted by

- [Magnus Tidemannn](https://github.com/Magnuti)
- [Jonas Klepper Rodningen](https://github.com/jonasrodningen)
- [Jørgen Hanssen](https://github.com/jorgenhanssen)

You can check out our [webpage](https://ntnuflightdelays.com) for an overview and an interactive demo.

## Setup

To install dependencies, simply run :

```txt
make setup
```

in the `root` directory of the project.

## Obtaining and Processing Data

We have provided a handful of flights and weather data between 2016 and 2019 as well as a complete trainingset for the data.
However, data can be obtained and processed in the following manner.

### Weather Data

To obtain weather data, navigate to `src/data/weather` and run the `scraper.py` script.
Then, run the `processor.py` to process that raw data.

### Flighs Records

Flight records are provided as is in the repository due to NDA restrictions. Contact [Avinor](https://avinor.no/en/) for a data agreement.
Then, run the `processor.py` script in `src/data/flights` to process the flight records.

### Creating the Dataset

When flight records and weather data are processed and exists in `data/processed`, run the `src/data/cleaner/main.py` script to merge the data into a final dataset.

## The ML Pipeline

In this project, we have a main pipeline for model experimentation and evaluation.
The pipeline can be found in `src/pipeline`.

### Pipeline Workflow

The pipeline has a folder `models` that contains seperated playgrounds for each method/model. The models are imported into the main program where it is provided data and run.
The run function inside each model allows for expirementation in the model's domain.

### Running the Pipeline

To run the pipeline, simply run the `main.py` inside the pipeline folder. To run a specific model playground, uncomment it from the main function.
