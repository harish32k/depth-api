
# Caption API

API to get a depth map from images, using the MiDAS depth estimation model. (open-source model)

## About
This is a Flask based web-application that is developed for deployment on Google Cloud Platform. This repository was successfully deployed to Google Cloud's Vertex AI service.
### Purpose
This application helps get a depth map of an image when an input image is given. The brightness of the image pixels will tell about the depth of the image. For more information, please refer to the paper of the authors of the model. [[1]](https://doi.org/10.1109/TPAMI.2020.3019967)

## How to build and deploy
- Build the docker image using the `Dockerfile` in the repository, preferably using a service like Google Cloud Build to make the process easy. 
- Deploy the model on Google Cloud Vertex AI platform using the docker image that is built.

MiDAS model link:
https://github.com/isl-org/MiDaS

More about the MiDAS model

1. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2022). Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(3), 1623–1637. https://doi.org/10.1109/TPAMI.2020.3019967

2. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision transformers for dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 12179-12188).
