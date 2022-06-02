# Satellite Image Segmention

This project is my exploration of semantic segmentation on satellite imagery.

## Problem Statement

The goal of the project is to segment satellite imagery into different classes of objects, i.e. woodlands, buildings, roads, etc. 

Potential applications include: 
- Monitor and quantify the changing state of our planet
- Reduce manual work of tracing over satellite imagery in map production

## Dataset

The dataset consists of 41 images and their corresponding masks obtained from: https://landcover.ai.linuxpolska.com/. The landcover consists of rasters three-channel GeoTiffs with EPSG:2180 spatial reference system while masks are single-channel. 

The masks contains 5 labels: unlablled (0), building (1), woodland (2), water(3), road(4). Unlablled class consists of all landcover types other than the types specified in labels 1-4. Overall, More than half of all mask pixels are unlablled, more than 1/3 are woodlands, and only a minority are buildings, water and road.

The images and masks were split into 256x256 smaller images. Image augmentation where additional datasets were created by randomly varying the brightness and constrast of source images.

## Modelling

2 segmentation architectures were used. A basic encoder-decorder network and Unet (https://arxiv.org/pdf/1505.04597.pdf). The basic network achieved 0.72 of IOU (Intersection over Union) and the Unet achieved 0.74 after about 30-40 epoches trained on the Google Colab platform. 

## Limitations and Future Exploration

### Limitations

- Does not handle overlapped objects well, i.e. tree over road
- Does not pick up areas where separation between objects are not as apparent
- Unlabelled class consists of grassland, concrete paving, shadow, etc.
- Some ground truth labelling are inaccurate
- Dataset only consists of some areas in Poland, mostly forested and rural areas

### Future explorations

- Explore other architectures and pretrained models
- Increase variety of datasets to include a fair representation of urban, rural, and different types of natural landcovers 
- Better labelled training sets


## Acknowledgement

The Unet architecture was based on the paper by Olaf Ronneberger, Philipp Fischer, and Thomas Brox: https://arxiv.org/pdf/1505.04597.pdf. 

In addition, the project takes reference from videos by DigitalSreeni: https://www.youtube.com/c/DigitalSreeni.


```python

```
