# SinGAN

[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) 
### Official pytorch implementation of the paper: "SinGAN: Learning a Generative Model from a Single Natural Image", ICCV 2019 Best paper award (Marr prize)


### Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{rottshaham2019singan,
  title={SinGAN: Learning a Generative Model from a Single Natural Image},
  author={Rott Shaham, Tamar and Dekel, Tali and Michaeli, Tomer},
  booktitle={Computer Vision (ICCV), IEEE International Conference on},
  year={2019}
}
```

## Code

### Install dependencies

```
python -m pip install -r requirements.txt
```

This code was tested with python 3.6

###  Train
To train SinGAN model on your own image, put the desire training image under Input/Images, and run

```
python main_train.py --input_name xxx.tif --input_dir xxx --nc_im 1 --nc_z 1 --max_size xxx
```

To run this code on a cpu machine, specify `--not_cuda` when calling `main_train.py`

###  Random samples
To generate random samples from any starting generation scale, please first train SinGAN model for the desire image (as described above), then run 

```
python sample.py --input_name xxx.tif --input_dir xxx --nc_im 1 --nc_z 1 --max_size xxx --not_cuda --gen_start_scale 0
```

pay attention: for using the full model, specify the generation start scale to be 0, to start the generation from the second scale, specify it to be 1, and so on. 

###  Random samples of arbitrery sizes
To generate random samples of arbitrery sizes, please first train SinGAN model for the desire image (as described above), then run 

```
python random_samples.py --input_name <training_image_file_name> --mode random_samples_arbitrary_sizes --scale_h <horizontal scaling factor> --scale_v <vertical scaling factor>
```

### Single Image Fréchet Inception Distance (SIFID score)
To calculate the SIFID between real images and their corresponding fake samples, please run:
```
python SIFID/sifid_score.py --path2real <real images path> --path2fake <fake images path> 
```  
Make sure that each of the fake images file name is identical to its cooresponding real image file name. Images should be saved in `.jpg` format.

### Super Resolution Results
SinGAN's SR results on BSD100 dataset can be download from the 'Downloads' folder.

### User Study
The data used for the user study can be found in the Downloads folder. 

real folder: 50 real images, randomly picked from the [places databas](http://places.csail.mit.edu/)

fake_high_variance folder: random samples starting from n=N for each of the real images 

fake_mid_variance folder: random samples starting from n=N-1 for each of the real images 

For additional details please see section 3.1 in our [paper](https://arxiv.org/pdf/1905.01164.pdf)


