### Keras-to-CSharp

#### Introduction

- It's a base class library of keras based on CSharp used to call. H5 simple deep learning model without any other environment.
- It could deploy your deep-learning model in C# without any other environment.
- The call to the small model (eg. BP net/shallow CNN) is much faster than tensorflow.

#### Advantage

- Easy extraction keras model
- Agile development
- Quick deployment
- Code readable

#### How to use

I make a [video](https://www.bilibili.com/video/av93374622) on bilibili to show you how to use in detail.

However,you can still use it by following these steps:

1. Open the script folder, put the. H5 model file in the **Input** folder (make sure it is a pure BP network structure or CNN structure), then run the **model_analysis.Py** file, and the weight file will be output in the **Output** folder.

2. Put 6 class library files in the program root directory.

3. Reference **Wineforever.Neuralnet.dll**.

4. Reference namespace:

   ```c#
   using Wineforever.Neuralnet;
   ```

5. Create a neural network **System**:

   ```C#
   var sys = new Wineforever.Neuralnet.System("sys");
   ```

6. Create **Layers**:

   ```C#
   //BP net
   sys.Create(512, "", "relu");  //Input Layer with 512 neurons and relu activation function
   sys.Create(256, "", "relu");  //Hidden Layer
   sys.Create(10, "", "softmax");  //Output Layer With softmax function
   
   //Or you can create CNN,for example:
   sys.Create(3,5); //Conv Layer with 3 kernels and the size is 5x5
   sys.Create(6,3);
   sys.Create("flatten"); //Flatten Layer
   sys.Create(128); //Dense Layer(Fully Connected)
   sys.Create(10,"","softmax");//Output Layer
   
   ```

7. **Load** Weights:

   ```C#
   sys.Load("d:\\myweights\\");//weights folder path
   ```

8. Set **Input**:

   ```C#
   sys.Input(train_x);
   ```

9. **Predict**:

   ```
   var res = sys.GetOutput();
   ```

With this class library,you can call Predict faster than using tf or keras.(I mean BP or CNN),for example,a BP net to predict mnist with 32 inputs and 10 outputs will be called for only 2-3ms.

I implement it through object-oriented method,each unit like neuron or kernel is individually encapsulated,and its bias implementation is exactly as same as Keras,so anyone who wants to get started with machine learning can learn from it.

#### Agreement

This class library follows the open source protocol and is not allowed to be used commercially.

#### Donate

If you want to sponsor me, you can scan the QR code below. Thank you for your support anyway.

![](http://106.15.93.194/donate/donate.PNG)

