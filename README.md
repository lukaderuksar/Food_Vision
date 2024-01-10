# Food_Vision_101 

This is an **end-to-end CNN Image Classification Model** built on the Tranfer Learning technique which identifies the food in your image.

A pretrained Image Classification Model EfficientnetB1 was retrained on the infamous **Food101 Dataset**.

The Model actually beats the [**DeepFood**](https://arxiv.org/abs/1606.05675) Paper's model which had an Accuracy of  was **77.4%**  also trained on the same dataset but the interesting thing is, DeepFood's model took 2-3 days to train while our's was around 60min.

> **Dataset :** `Food101`

> **Model :** `EfficientNetB1`
We're going to be building Food Vision Big™, using all of the data from the Food101 dataset.

Yep. All 75,750 training images and 25,250 testing images.

Two methods to significantly improve the speed of our model training:

Prefetching
Mixed precision training

### **Checking the GPU**

For this Project we will working with **Mixed Precision**. And mixed precision works best with a with a GPU with compatibility capacity **7.0+**.

At the time of writing, colab offers the following GPU's :
* Nvidia K80
* **Nvidia T4**
* Nvidia P100

Colab allocates a random GPU everytime we factory reset runtime. So you can reset the runtime till you get a **Tesla T4 GPU** as T4 GPU has a rating 7.5.

> In case using local hardware, use a GPU with rating 7.0+ for better results.


## **Preprocessing the Data**

Since we've downloaded the data from TensorFlow Datasets, there are a couple of preprocessing steps we have to take before it's ready to model. 

More specifically, our data is currently:

* In `uint8` data type
* Comprised of all differnet sized tensors (different sized images)
* Not scaled (the pixel values are between 0 & 255)

Whereas, models like data to be:

* In `float32` data type
* Have all of the same size tensors (batches require all tensors have the same shape, e.g. `(224, 224, 3)`)
* Scaled (values between 0 & 1), also called normalized

To take care of these, we'll create a `preprocess_img()` function which:

* Resizes an input image tensor to a specified size using [`tf.image.resize()`](https://www.tensorflow.org/api_docs/python/tf/image/resize)
* Converts an input image tensor's current datatype to `tf.float32` using [`tf.cast()`](https://www.tensorflow.org/api_docs/python/tf/cast)

## **Building the Model : EfficientNetB1**

Implemented Mixed Precision training and Prefetching to decrease the time taken for the model to train.

### **Getting the Callbacks ready**
As we are dealing with a complex Neural Network (EfficientNetB0) its a good practice to have few call backs set up. Few callbacks I will be using throughtout this Notebook are :
 * **TensorBoard Callback :** TensorBoard provides the visualization and tooling needed for machine learning experimentation

 * **EarlyStoppingCallback :** Used to stop training when a monitored metric has stopped improving.
 
 * **ReduceLROnPlateau :** Reduce learning rate when a metric has stopped improving.
 
### Evaluating the results
 
### Loss vs Epochs
 
![image](https://user-images.githubusercontent.com/61462986/202082223-83c3a8f2-26c9-455e-97d5-ee833a4b10cc.png)

### Accuracy vs Epochs

![image](https://user-images.githubusercontent.com/61462986/202082253-0d28ea8e-72af-4182-bf79-33b4119f27ef.png)

### Model's Class-wise Accuracy Score

 ![image](https://user-images.githubusercontent.com/61462986/202082047-6690d7cd-1999-4edc-9dc1-53fb9780ee89.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/61462986/202082179-3337c5d7-fa06-4589-9050-1c2af1785808.png)

### Custom Prediction

![image](https://user-images.githubusercontent.com/61462986/202090045-4469ad2b-5366-41ed-8abc-1e5013b55ae6.png)



