# Style transfer
## Intro
A basic style transfer Neural Network to experiment with using transfer learning.

Initial goal is to see how the NN transfers my style of painting and Cezanne's style of painting to a photograph which I recently painted twice, once in my typical style and once using Cezanne's paintings as inspiration.

## Setup

Poetry is used for version control.

Once poetry is installed, run
```shell
poetry install
```
to install all packages used.

To generate an image with the content of image1 and a style of image2 using poetry, run
```shell
poetry run python .\src\style-transfer\main.py
```

