## AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022, Official Code)

[Jiaxi Jiang<sup>1</sup>](https://www.jiaxi-jiang.com/), [Paul Streli<sup>1</sup>](https://www.paulstreli.com), Huajian Qiu<sup>1</sup>, [Andreas Fender<sup>1</sup>](http://www.andreasfender.com/), Larissa Laich<sup>2</sup>, [Patrick Snape<sup>2</sup>](http://patricksnape.github.io/), [Christian Holz<sup>1</sup>](https://www.christianholz.net)<br/>

<sup>1</sup> [Sensing, Interaction & Perception Lab](https://siplab.org), Department of Computer Science, ETH Zürich, Switzerland <br/>
<sup>2</sup> Meta Reality Labs, Zurich, Switzerland <br/>

___________

Contents
----------

* [Datasets](#datasets)
* [Training](#training)
* [Testing](#testing)
* [Pretrained Models](#pretrained-models)
* [Citation](#citation)
* [License and Acknowledgement](#license-and-acknowledgement)

Datasets
----------
1. Please download the datasets `BMLrub`, `CMU`, and `HDM05` from [AMASS](https://amass.is.tue.mpg.de/index.html).
2. Run `prepare_data.py` to preprocess the input data for faster training. The data split for training and testing data is stored under the folder `data_split`.

Training
----------
For training, please run:
```
python main_train_avatarposer.py -opt options/train_avatarposer.json
```
Testing
----------
For testing, please run:
```
python main_test_avatarposer.py
```
Pretrained Models
----------
Click [Pretrained Models]() to download our pretrained model for AvatarPoser.

Citation
----------
If your find our paper or codes useful, please cite our work:

    @inproceedings{jiang2022avatarformer,
      title={AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing},
      author={Jiang, Jiaxi and Streli, Paul and Qiu, Huajian and Fender, Andreas and Laich, Larissa and Snape, Patrick and Holz, Christian},
      booktitle={Proceedings of European Conference on Computer Vision},
      year={2022},
      organization={Springer}
    }

License and Acknowledgement
----------
