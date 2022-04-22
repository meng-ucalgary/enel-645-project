# ENEL-645 Winter-2022 Group-1 Prof. Roberto Souza

This repository contains the code for the final project of the ENEL645 course

## Folder structure

This is a brief description of the contents of each folder:

- [dataset](dataset): dataset used for training and validation.

- [dataset_test](dataset_test): dataset used for testing, created by extracting 20% of the images of the training/validation dataset.

- [dataset_test_easy](dataset_test_easy): a small dataset consisting of non-covered cards that are near the camera, those images where augmented with rotation. Meant to test the model in a almost ideal scenario.

- [dataset_test_hard](dataset_test_hard): a small dataset consisting of partially covered cards far from the camera, with no augmentation applied. Meant to test the model in a extreme scenario.

- [custom_model](custom_model): code used to train and test the custom model created. For both single and two outputs.

- [EfficientNet](EfficientNet): code used to train and test the EfficientNet architecture from no pre defined weights.

- [Resnet50](Resnet50): code used to train and test the Resnet50 architecture from no pre defined weights.

- [VGG16](VGG16): code used to train and test the VGG16 architecture from no pre defined weights.

- [Transfer_Learning](Transfer_Learning): code used to train the aforementioned architectures (but the custom model) using Transfer Learning.

- [split_code](split_code): code used to split the test dataset from the train/validation dataset.

- [sample_apps](sample_apps): contains two sample apps that uses the best models obtained (EfficientNet B4)
  - [calculate_poker_odds](sample_apps/calculate_poker_odds/): takes the image of a poker table and a model as input, extracts the cards from the image, uses the model to classify those cards, and calculates the odds of winning for players. Check the README.md for more details on how to run
  - [sample_video](sample_apps/sample_video/): contains a pre-recorded video of cards being classified in real time from frames obtained from a video stream. Also the code to run the video on a Jetson Nano board. Check the README.md for more detials on how to run

## Contributors

- [Bhavyai Gupta](https://github.com/zbhavyai)
- [Drew Burritt](https://github.com/dburritt)
- [Jose Carlos Cazarin Filho](https://github.com/espiriki)
- [Michael Man Yin Lee](https://github.com/mlee2021)
- [Thomas Scott](https://github.com/tscott6)
