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

#### Agreement

This class library follows the open source protocol and is not allowed to be used commercially.

#### Donate

If you want to sponsor me, you can scan the QR code below,or donate bitcoin to me. Thank you for your support anyway.

Bitcoin Wallet:16RpsEY6C1zLZTPZUX8mXK9ozooqhh5YqS

![](http://106.15.93.194/donate/donate.png)
