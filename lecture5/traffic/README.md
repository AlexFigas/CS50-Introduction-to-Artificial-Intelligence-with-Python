# AlexFigas
## Traffic
The convolutional neural network model is based on the TensorFlow Keras sequential model.

To learn the data set, I applied a convolutional layer of 32 filters and a 3x3 kernel, a maximum grouping layer of 2x2 units, and a hidden layer of 128 units. Consequently, it was insufficient for learning the data set. A convolutional layer and a grouping layer probably did not generalize the image enough, and a hidden layer was certainly insufficient for such a complex dataset.

A second hidden layer of 128 units should be added with the same reluctance. Although the results were much better, there was still room for improvement. The model train was improved by having more layers.

# How to run:

0. Install the dependencies: Python >= 3 and requirements (```pip install -r requirements.txt```)
1. Download the [Dataset](https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip) and unzip it in the same folder as the code.
2. Run the code in the terminal with the command: ```python traffic.py gtsrb```
3. The program will train the model and then test it. (It will take a while)
  
Click here to watch the video of the program running: [Video](https://youtu.be/UresboKBH4Y)

![Screenshot](cap.png)