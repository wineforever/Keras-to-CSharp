### Keras-to-CSharp

#### Introduction

- It's a base class library of keras based on C# used to call .H5 simple deep learning model.
- It could deploy the deep-learning model in your application without any other environment.

#### Advantage

- Easy extraction keras model
- Agile development
- Quick deployment
- Code readable

#### How to use

I make a [video](https://www.bilibili.com/video/av93374622) on bilibili to show you how to use in detail.

However, you can still learn about it by following these steps:

1. Open the script folder, put the. H5 model file in the **Input** folder (make sure it is a pure BP network structure or CNN structure), then run the **model_analysis.py** file, press **Enter** all the way, and the weight file will be output in the **Output** folder.

2. Put class library files in the program root directory.

3. Import **Wineforever.Neuralnet.dll** and reference namespace:

   ```c#
   using Wineforever.Neuralnet;
   ```

5. Create a neural network **System**:

   ```C#
   var sys = new Wineforever.Neuralnet.System("sys");
   ```

6. Create **Layers**, make sure the model structure is consistent:

   ```C#
   //BP net
   sys.Create(512, "", "relu");  //512 neurons with relu activation function
   sys.Create(256, "", "relu");
   sys.Create(10, "", "softmax");
   
   //Or you can create CNN,for example:
   sys.Create(3,5);  //3 kernels and the size is 5x5
   sys.Create(6,3);
   sys.Create("flatten");
   sys.Create(128);
   sys.Create(10,"","softmax");
   ```
   
6. **Load** Weights:

   ```C#
   sys.Load(@"d:\myweights\");  //weights folder path
   ```

7. Set **Input**:

   ```C#
   sys.Input(data);
   ```

8. **Predict**:

   ```C#
   var res = sys.GetOutput();
   ```

With this class library,you can call Predict faster than using tf or keras.(BP or CNN),for example,a BP net to predict mnist with 32 inputs and 10 outputs will be called for only 2-3ms.

I implement it through object-oriented method,each unit like neuron or kernel is individually encapsulated,and its bias implementation is exactly as same as Keras,so anyone who wants to get started with machine learning can learn from it.

#### What's New?

In Version 2.0, I simplified the neural network structure and greatly improved the forward propagation speed.

Because I found that the reason for the slow speed of the old version is that it excessively encapsulates neurons and convolution kernels (in order to make the back propagation easier), but after I decided to abandon the back propagation, I reconstructed the structure of the dense layer and the convolution layer. Now they have actually become a one-time matrix operation, and the efficiency has been greatly improved.

Update Log:

- V 2.0:

  1) The network structure has been optimized, the forward propagation speed has been greatly improved, and the efficiency has been increased by 1000 times.

  2) It focuses on the forward propagation of neural network and no longer supports back propagation.

- V 1.3:Without reference to Lua script, it no longer supports custom interface, which makes class library lightweight.

- V 1.2:

  1) Added the volume accumulation layer and flatten layer.

  2) Load function is added to support reading model weight parameters.

  3) Add loadkernel, loadbias, print and other methods for layer.

- V 1.1: 

  1) Now the input layer also has neurons, and the number of neurons in the input layer does not have to be the same as the number of data in the input dataset.

  2) When you update the system now, you will not delete the neurons in the first layer and create them again, but just change their connection.

  3) Fixed the error that the weight was no longer initialized after the network layer was added

  4) The MSE loss function is corrected

- V 1.0: The basic framework of neural network model is established.

#### Agreement

This class library follows the open source protocol and is not allowed to be used commercially.

#### Donate

If you want to sponsor me, you can scan the QR code below,or donate bitcoin to me. Thank you for your support anyway.

Bitcoin Wallet:16RpsEY6C1zLZTPZUX8mXK9ozooqhh5YqS

![](http://106.15.93.194/donate/donate.png)
