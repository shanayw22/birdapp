
# Mini-project-2: Bird classification application

**Goal**: Develop a toy version of the Merlin Bird App by building from scratch both an audio classifier (using spectrograms obtained form audio files with bird songs) and an image classifier to identify bird species, then deploy a user application with Streamlit (or similar).  

For the project, please try to build each component as much from scratch as possible, i.e pretend you are the original research team who made the Merlin app and try to recreate something similar from scratch. Obviously you can wrap around large popular libraries like PyTorch, but please try to do as much as possible "on your own".  

## Project Summary

### Group selection

You can work alone if you want but recommended to work as a groupd

- **Form a group of 3 people**
  - (Optional): Select a group leader
  - Plan out your project in a systematic way 
  - Break the project into stages and allocate sub-components to individuals 
  - Feel free to use Cursor, or similar, for coding acceleration 
  - Use Git, branches, etc to collaborate professionally, and efficiently 

### Reading & Research:

To learn more about what the Merlin App is, see the following link:

* https://jfh.georgetown.domains/centralized-lecture-content/content/data-science/introductory-content/merlin-demo/

Additional useful information can also be found at the following:

- Review the following articles to understand spectrogram creation and the inner workings of Merlin’s Sound ID:
   - [From Sound to Images, Part 1: A Deep Dive on Spectrogram Creation](https://www.macaulaylibrary.org/2021/07/19/from-sound-to-images-part-1-a-deep-dive-on-spectrogram-creation/)
   - [From Sound to Images, Part 2: Spectrogram Image Processing](https://www.macaulaylibrary.org/2021/08/05/from-sound-to-images-part-2-spectrogram-image-processing/)
   - [Behind the Scenes of Sound ID in Merlin](https://www.macaulaylibrary.org/2021/06/22/behind-the-scenes-of-sound-id-in-merlin/)
- Familiarize yourself with the guide on bird sound downloads: [Guide to Bird Sounds – Download Info](https://www.macaulaylibrary.org/guide-to-bird-sounds/download-info/)
  

### Data Acquisition:

- Start simple!!! As always, use a simple, small & fast dataset to debug and develop your pipeline. Once the project is working for a small dataset, then apply the workflow to a more complex, real-world dataset chosen by your group 


`Toy data`

There are larger datasets, which you can consider later for "beefing up" your model, however to start here are two relatively "small" data sets for prototyping and debugging 

* Bird images: https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images
* Bird sounds: https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set

These can be downloaded with the following code 

```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("vinayshanbhag/bird-song-data-set")

print("Path to dataset files:", path)

# move path to current folder
import shutil
shutil.move(path, ".")
```


```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("wenewone/cub2002011")

print("Path to dataset files:", path)

# move path to current folder
import shutil
shutil.move(path, ".")
```

`Taking it further:` 

I could be wrong, but I suspect you will be able to get the previous data sets working relatively quickly. If that is the case, consider "scaling up" and attempting to repeat the exercise using a larger data sets. e.g. The one used by the original Merlin app (or some reasonable subset of the dataset). 


### Preprocessing:

- **Audio:** Convert raw bird audio clips into spectrogram images to serve as inputs for your audio classifier.
- **Images:** Standardize image sizes (e.g., resize to 224×224) and perform necessary augmentations.
- As usual, have a training/test/validation split

### Model Training:

- **Audio Classification:** Train models (e.g., VGG and ResNet architectures) using spectrogram images to classify bird species from audio clips.
- **Image Classification:** Similarly, train VGG and ResNet models on bird images.
- Experiment with both transfer learning (pretrained models) and training from scratch where possible.
- Compare performance metrics (accuracy, loss, inference time) across models and modalities.
- Tune the models as needed for best results

### Deployment:

- Create an interactive Streamlit app that allows users to:
 - Upload an audio file to view its spectrogram and receive a predicted bird species.
 - Upload a bird image to receive a classification result.
- Ensure the app displays both prediction outputs and, optionally, model confidence scores.

### Documentation & Submission:

- Submit to the code repository with clear documentation and comments.
- Provide a brief report summarizing your approach, data preprocessing steps, model comparisons, challenges encountered, and final evaluation metrics.
* After two weeks, you will need to submit the following
	* A link to your GitHub code repository 
	* Make a slide deck (or jupyter notebook), to describe what you did and showcase your results (submit PDF or HTML to Canvas)
* Do a recording where you walk through and describe your results and presentation (10-20 minutes) (submit the recording to Canvas presentation recording)
	* at the end of the mini-project, a group will be selected at random to do their presentation "live" in front of class (to fuel discussion)


## Appendix

### Toy dataset information 

More information on the toy data sets 

#### Image

Caltech-UCSD Birds-200-2011 (CUB-200-2011) is an extended version of the CUB-200 dataset, with roughly double the number of images per class and new part location annotations.

Number of categories: 200
Number of images: 11,788
Annotations per image: 15 Part Locations, 312 Binary Attributes, 1 Bounding Box
For detailed information about the dataset, please see the technical report at https://authors.library.caltech.edu/27452/

#### Audio

Small dataset for testing and prototyping.

Data set includes only "songs" from 5 species-
- Bewick's Wren
- Northern Cardinal
- American Robin
- Song Sparrow
- Northern Mockingbird

For simplicity, data set excludes other types of calls (alarm calls, scolding calls etc) made by these birds. Additionally, only recordings graded as highest quality on xeno-canto API are included.

Original recordings from xeno-canto were in mp3. All files were converted to wav format. Further, using onset detection, original recordings of varying lengths were clipped to exactly 3 sec such that some portion of the target bird's song is included in the clip.

Original mp3 files from the source have varying sampling rates and channels. As part of the conversion process each wav file was sampled at exactly 22050 samples/sec and is single channel.

CSV file includes recording metadata, such as genera, species, location, datetime, source url, recordist and license information.
The filename column in CSV corresponds to the wav files under wavfiles folder
Acknowledgements
All information is sourced from API at https://www.xeno-canto.org/

**Inspiration** What features in a birds song are critical in distinguishing it from other species? How accurately can we identify a bird given a 3 s recording?

**Source**: https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set
