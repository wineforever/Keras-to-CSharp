using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LuaInterface;
using Wineforever.String;
namespace Wineforever.Neuralnet
{
    public class Neuron //定义神经元
    {
        //公共字段
        public Neuron(dynamic input = null, string name = null, string activation = "relu", string optimizer = "GD")//构造神经元
        {
            //---- 生成检索码 ----
            var GUID = Guid.NewGuid();//全局唯一标识符
            string ID = GUID.ToString().Replace("-", "").ToUpper();//转换成大写
            //---- ----
            //---- 初始化权重 ----
            if (input != null)
            {
                Type input_type = input.GetType();
                if (input_type == typeof(double))
                    Initialization(0);
                else if (input_type == typeof(List<Neuron>))
                    Initialization(input.Count);
                else if (input_type == typeof(List<double>))
                    Initialization(input.Count);
                else if (input_type == typeof(FlattenLayer))
                    Initialization(0);
            }
            else
                Initialization(0);
            //---- ----
            //---- 创建神经元 ----
            this.Name = name == null ? "Undefined" : name;
            this.ID = ID;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            //---- ----
        }
        //私有字段
        internal string Name { get; set; }//标识名
        internal string ID { get; }//检索码
        internal dynamic Input { get; set; }//多输入
        internal string Activation { get; }//激励器
        internal string Optimizer { get; }//优化器
        internal DenseLayer Father { get; set; }//父对象
        internal List<double> Weights { get; set; }//权重
        internal double Baisc { get; set; }//偏置量
        private bool __isHidden__ = false;
        //公共方法
        public double GetOutput()//返回当前神经元的输出
        {
            double output = 0;
            //---- 判断输入类型 ----
            Type input_type = this.Input.GetType();
            if (input_type == typeof(double))
            {
                return this.Input;
            }
            if (input_type == typeof(List<Neuron>))
            {
                if (this.Optimizer.Contains("dropout") && this.Father.Father.Rand(0, 1) <= this.Father.Father.dropout_rate)
                {
                    output = 0;
                    this.__isHidden__ = true;
                }
                else
                {
                    List<Neuron> input_neurons = this.Input;//得到输入神经元
                    for (int i = 0; i < input_neurons.Count; i++)
                    {
                        Neuron neuron = input_neurons[i];
                        var neuron_output = neuron.GetOutput();
                        output += neuron_output * this.Weights[i];
                    }
                    output += this.Baisc;
                }
            }
            else if (input_type == typeof(List<double>))
            {
                if (this.Optimizer.Contains("dropout") && this.Father.Father.Rand(0, 1) <= this.Father.Father.dropout_rate)
                {
                    output = 0;
                    this.__isHidden__ = true;
                }
                else
                {
                    List<double> input = this.Input;
                    for (int i = 0; i < input.Count; i++)
                    {
                        output += input[i] * this.Weights[i];
                    }
                    output += this.Baisc;
                }
            }
            else if (input_type == typeof(FlattenLayer))
            {
                if (this.Optimizer.Contains("dropout") && this.Father.Father.Rand(0, 1) <= this.Father.Father.dropout_rate)
                {
                    output = 0;
                    this.__isHidden__ = true;
                }
                else
                {
                    List<double> input = this.Input.GetOutput();
                    for (int i = 0; i < input.Count; i++)
                    {
                        output += input[i] * this.Weights[i];
                    }
                    output += this.Baisc;
                }
            }
            return activation(output, this.Activation);
        }
        public void Connect(Neuron neuron)//将当前神经元的输出端连接到指定神经元的输入端
        {
            List<Neuron> inputs = neuron.Input;
            inputs.Add(this);
            neuron.Input = inputs;
        }
        public void Disconnect(Neuron neuron)//切断与指定神经元的连接
        {
            List<Neuron> inputs = neuron.Input;
            inputs.Remove(this);
            neuron.Input = inputs;
        }
        public void Update(double error, string optimizer)//更新权重
        {
            var Weights = this.Weights;
            var Basic = this.Baisc;
            double learning_rate = this.Father.Father.learning_rate;
            List<double> deltas = new List<double>();
            if (!this.__isHidden__)
            {
                for (int i = 0; i < Weights.Count; i++)
                {
                    double input = 0;
                    string activation = "normal";
                    if (this.Input.GetType() == typeof(List<Neuron>))
                    {
                        input = this.Input[i].GetOutput();
                        activation = this.Input[i].Activation;
                    }
                    else if (this.Input.GetType() == typeof(List<double>))
                    {
                        input = this.Input[i];
                        activation = "normal";
                    }
                    double delta = 0;
                    //优化器
                    if (optimizer.Contains("GD"))//标准梯度下降法
                    {
                        delta = error * input * autodiff(this.GetOutput(), activation) * learning_rate;
                    }
                    else if (optimizer.Contains("BGD"))//批量梯度下降法
                    {
                        delta = this.Father.Father.__loss__ * input * autodiff(this.GetOutput(), activation) * learning_rate;
                    }
                    else if (optimizer.Contains("SGD"))//随机梯度下降法
                    {
                        double SGD = this.Father.Father.errors[(int)this.Father.Father.Rand(0, (this.Father.Father.Layers.Last() as DenseLayer).Neurons.Count)] * input * autodiff(this.GetOutput(), activation);
                        delta = learning_rate * SGD;
                        Console.WriteLine("SGD");
                    }
                    else//自定义梯度下降实现
                    {
                        double SGD = this.Father.Father.errors[(int)this.Father.Father.Rand(0, (this.Father.Father.Layers.Last() as DenseLayer).Neurons.Count)] * input * autodiff(this.GetOutput(), activation);
                        string location = AppDomain.CurrentDomain.BaseDirectory + "optimizers\\" + optimizer + ".lua";
                        Lua lua = new Lua();
                        lua.DoString("lr=" + learning_rate);
                        lua.DoString("err=" + error);
                        lua.DoString("y=" + input);
                        lua.DoString("diff=" + autodiff(this.GetOutput(), activation));
                        lua.DoString("loss=" + this.Father.Father.__loss__);
                        lua.DoString("rand=" + this.Father.Father.errors[(int)this.Father.Father.Rand(0, (this.Father.Father.Layers.Last() as DenseLayer).Neurons.Count)]);
                        lua.DoString("SGD=" + SGD);
                        lua.DoString(Wineforever.String.client.Load(location));
                        delta = (double)lua.DoString("return dw")[0];
                    }
                    deltas.Add(delta);
                    //---- 更新权重 ----
                    Weights[i] -= deltas[i];
                    //---- ----
                }
                //---- 更新偏置量 ----
                Basic -= error * learning_rate;
                //---- ----
                this.Weights = Weights;
                this.Baisc = Baisc;
                //---- 找到前级神经元，递归更新权重 ----
                if (this.Input.GetType() == typeof(List<Neuron>))
                {
                    var pre_layer_neurons = this.Input;
                    for (int i = 0; i < pre_layer_neurons.Count; i++)
                        if (pre_layer_neurons[i].Input.GetType() == typeof(List<Neuron>) || pre_layer_neurons[i].Input.GetType() == typeof(List<double>))
                            pre_layer_neurons[i].Update(deltas[i], pre_layer_neurons[i].Optimizer);
                }
            }
            else { this.__isHidden__ = false; }//取消隐藏
        }
        public void Save()//输出权重信息
        {
            string FileName = AppDomain.CurrentDomain.BaseDirectory + "neurons\\" + this.Name + ".txt";
            Dictionary<string, string> Data = new Dictionary<string, string>();
            Data["Name"] = this.Name;
            Data["ID"] = this.ID;
            Data["Activation"] = this.Activation;
            Data["Optimizer"] = this.Optimizer;
            var neurons = this.Input;
            for (int i = 0; i < neurons.Count; i++)
            {
                Data["Weight" + i] = this.Weights[i].ToString();
            }
            Data["Basic"] = this.Baisc.ToString();
            Wineforever.String.client.SaveToList(Data, FileName);
        }
        public void SetWeight(double Value)
        {
            for (int i = 0; i < this.Weights.Count; i++)
            {
                this.Weights[i] = Value;
            }
        }
        public void SetWeight(List<double> Values)
        {
            for (int i = 0; i < this.Weights.Count; i++)
            {
                this.Weights[i] = Values[i];
            }
        }
        public void SetWeight(List<string> Values)
        {
            for (int i = 0; i < this.Weights.Count; i++)
            {
                this.Weights[i] = double.Parse(Values[i]);
            }
        }
        public void SetBias(double Value)
        {
            this.Baisc = Value;
        }
        public void SetBias(string Value)
        {
            this.Baisc = double.Parse(Value);
        }
        public void Print()
        {
            for (int i = 0; i < this.Weights.Count; i++)
                Console.WriteLine("weights:{0}", this.Weights[i]);
            for (int i = 0; i < this.Input.Count; i++)
                Console.WriteLine("input:{0}", this.Input[i]);
            Console.WriteLine("bias:{0}", this.Baisc);
            Console.WriteLine("output:{0}", this.GetOutput());
        }
        //私有方法
        //参数初始化
        internal void Initialization(int input_Count)
        {
            var Weights = new List<double>();
            for (int i = 0; i < input_Count; i++)
            {
                Weights.Add(1);
            }
            this.Weights = Weights;
            this.Baisc = 0;
        }
        //调用外部激励器
        private double activation(double x, string func)
        {
            if (func == "relu") return x > 0 ? x : 0;
            else if (func == "normal") return x;
            else if (func == "sigmoid") return 1.0 / (1.0 + Math.Pow(Math.E, -x));
            else if (func == "leaky relu") return x > 0 ? x : 0.01 * x;
            else if (func == "softmax") return x;
            else
            {
                string location = AppDomain.CurrentDomain.BaseDirectory + "activations\\" + func + ".lua";
                Lua lua = new Lua();
                lua.DoString("x=" + x);
                lua.DoString(Wineforever.String.client.Load(location));
                var res = (double)lua.DoString("return y")[0];
                return res;
            }
        }
        //自动求导
        private double autodiff(double x, string func)
        {
            if (func == "relu") return x > 0 ? 1 : 0;
            else if (func == "normal") return 1;
            else if (func == "sigmoid") return activation(x, "sigmoid") * (1 - activation(x, "sigmoid"));
            else if (func == "leaky relu") return x > 0 ? 1 : 0.01;
            else if (func == "softmax") return 1;
            else
            {
                double d = 1e-4;//数值微分
                return (activation(x + d, func) - activation(x, func)) / d;
            }
        }
        //---- ----
    }
    public class Kernel//定义卷积核
    {
        //公共字段
        public Kernel(int size,dynamic input = null,string name = null,string padding = "same",int step = 1,string activation = "relu",string optimizer = "GD")//构造卷积核
        {
            //---- 生成检索码 ----
            var GUID = Guid.NewGuid();//全局唯一标识符
            string ID = GUID.ToString().Replace("-", "").ToUpper();//转换成大写
            //---- ----
            //---- 初始化权重 ----
            if (input != null)
            {
                Type input_type = input.GetType();
                if (input_type == typeof(double[,]))
                {
                    Initialization(size, 1);
                }
                else if (input_type == typeof(List<double[,]>))
                {
                    Initialization(size, input.Count());
                }
                else if (input_type == typeof(ConvLayer))
                {
                    Initialization(size, (input as ConvLayer).Kernels.Count());
                }
            }
            else
            {
                Initialization(size, 4);
            }
            //---- ----
            //---- 创建卷积核 ----
            this.Name = name == null ? "Undefined" : name;
            this.ID = ID;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            this.Size = size;
            this.Padding = padding;
            this.Step = step;
            //---- ----
        }
        //私有字段
        internal string Name { get; set; }//标识名
        internal string ID { get; }//检索码
        internal dynamic Input { get; set; }//多输入
        internal string Activation { get; }//激励器
        internal string Optimizer { get; }//优化器
        internal ConvLayer Father { get; set; }//父对象
        internal double[,,] Weights { get; set; }//权重
        internal double Bias { get; set; }//偏置量
        internal int Size { get; set; }//尺寸
        internal string Padding { get; }//扩充模式
        internal int Step { get; set; }//步长
        //公有方法
        public dynamic GetOutput()
        {
            dynamic output = null;
            // --- 判断输入类型 --- 
            Type input_type = this.Input.GetType();
            if (input_type == typeof(double[,]))
            {
                //单通道
                output = Wineforever.Mathematics.client.conv(Input, Weights, Bias, Step, Activation, Padding);
                return output;
            }
            else if (input_type == typeof(List<double[,]>))
            {
                //多通道
                output = Wineforever.Mathematics.client.conv(Input, Weights, Bias, Step, Activation, Padding);
                return output;
            }
            else if (input_type == typeof(ConvLayer))
            {
                var input = this.Input.GetOutput();
                if (input.GetType() == typeof(double[,]))
                {
                    output = Wineforever.Mathematics.client.conv(input, Weights, Bias, Step, Activation, Padding);
                }
                else if (input.GetType() == typeof(List<double[,]>))
                {
                    output = Wineforever.Mathematics.client.conv(input, Weights, Bias, Step, Activation, Padding);
                }
                return output;
            }
            return -1;
        }
        public void SetWeight(double Value)
        {
            int width = Weights.GetLength(0);
            int height = Weights.GetLength(1);
            int channel = Weights.GetLength(2);
            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    for (int k = 0; k < channel; k++)
                        Weights[i, j, k] = Value;
        }
        public void SetWeight(double[,,] Values)
        {
            int width = Weights.GetLength(0);
            int height = Weights.GetLength(1);
            int channel = Weights.GetLength(2);
            for (int i = 0; i < width; i++)
                for (int j = 0; j < height; j++)
                    for(int k=0;k<channel;k++)
                    Weights[i, j,k] = Values[i, j,k];
        }
        public void SetWeight(double Value, int width_index, int height_index, int channel_index)
        {
            Weights[width_index, height_index, channel_index] = Value;
        }
        public void SetBias(double Value)
        {
            Bias = Value;
        }
        //私有方法
        internal void Initialization(int size,int channel)
        {
            var Weights = new double[size, size,channel];
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    for (int k= 0; k < channel; k++)
                        Weights[i, j,k] = 1.0;
            this.Weights = Weights;
            this.Bias = 0;
        }
        //调用外部激励器
        private double activation(double x, string func)
        {
            if (func == "relu") return x > 0 ? x : 0;
            else if (func == "normal") return x;
            else if (func == "sigmoid") return 1.0 / (1.0 + Math.Pow(Math.E, -x));
            else if (func == "leaky relu") return x > 0 ? x : 0.01 * x;
            else if (func == "softmax") return x;
            else
            {
                string location = AppDomain.CurrentDomain.BaseDirectory + "activations\\" + func + ".lua";
                Lua lua = new Lua();
                lua.DoString("x=" + x);
                lua.DoString(Wineforever.String.client.Load(location));
                var res = (double)lua.DoString("return y")[0];
                return res;
            }
        }
    }
    //---- 层类 ----
    public abstract class Layer
    {
        //私有字段
        internal string Name { get; set; }//标识名
        internal dynamic Input { get; set; }//输入
        internal string Activation { get; set; }
        internal string Optimizer { get; set; }
        internal System Father { get; set; }
        //公有方法
        public dynamic GetOutput()
        {
            if (this.GetType() == typeof(DenseLayer))
            {
                return (this as DenseLayer).GetOutput();
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                return (this as ConvLayer).GetOutput();
            }
            else if (this.GetType() == typeof(FlattenLayer))
            {
                return (this as FlattenLayer).GetOutput();
            }
            return null;
        }
        public void LoadKernel(string FileName)
        {
            if (this.GetType() == typeof(DenseLayer))
            {
                (this as DenseLayer).LoadNeuron(FileName);
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                (this as ConvLayer).LoadKernel(FileName);
            }
        }
        public void LoadBias(string FileName,int Format =0 )
        {
            if (this.GetType() == typeof(DenseLayer))
            {
                (this as DenseLayer).LoadBias(FileName);
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                (this as ConvLayer).LoadBias(FileName, Format);
            }
        }
        public void Print()
        {
            Console.WriteLine("=======================");
            Console.WriteLine("========= {0} =========",Name);
            Console.WriteLine("=======================");
            Console.WriteLine("Input: {0}", Input);
            Console.WriteLine("Activation: {0}", Activation);
            Console.WriteLine("Optimizer: {0}", Optimizer);
            Console.WriteLine("Father: {0}", Father);
            if (this.GetType() == typeof(DenseLayer))
            {
                for (int i = 0; i < (this as DenseLayer).Neurons.Count(); i++)
                {
                    Console.WriteLine("---- {0} ----", (this as DenseLayer).Neurons[i].Name);
                    Wineforever.Mathematics.client.print((this as DenseLayer).Neurons[i].Weights);
                    Console.WriteLine("Bias: {0}",(this as DenseLayer).Neurons[i].Baisc);
                }
            }
            else if (this.GetType() == typeof(ConvLayer))
            {
                Console.WriteLine("Kernel: ");
                for (int i = 0; i < (this as ConvLayer).Kernels.Count(); i++)
                {
                    Console.WriteLine("---- {0} ----", (this as ConvLayer).Kernels[i].Name);
                    Wineforever.Mathematics.client.print((this as ConvLayer).Kernels[i].Weights);
                    Console.WriteLine("Bias: {0}", (this as ConvLayer).Kernels[i].Bias);
                }
            }
        }
    }
    //全连接层
    public class DenseLayer : Layer
    {
        //私有字段
        internal string __kernelPath__ = "";
        internal string __biasPath__ = "";
        //公有字段
        public DenseLayer(int neuron_num, dynamic input = null, string name = null, string activation = "relu", string optimizer = "GD")
        {
            //---- 创建层 ----
            this.Name = name == null ? "Undefined" : name;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            //---- ----
            //---- 创建神经元 ----
            this.Neurons = new List<Neuron>();
            //---- ----
            //判断输入类型
            if (input != null)
            {
                if (input.GetType() == typeof(DenseLayer))//非头层
                {
                    DenseLayer Input = input;
                    for (int i = 0; i < neuron_num; i++)
                    {
                        Neuron neuron = new Neuron(Input.Neurons, name + "(" + i + ")", activation, optimizer);
                        neuron.Father = this;//设置父对象
                        this.Neurons.Add(neuron);
                    }
                }
                if (input.GetType() == typeof(FlattenLayer))
                {
                    FlattenLayer Input = input;
                    for (int i = 0; i < neuron_num; i++)
                    {
                        Neuron neuron = new Neuron(Input, name + "(" + i + ")", activation, optimizer);
                        neuron.Father = this;//设置父对象
                        this.Neurons.Add(neuron);
                    }
                }
                else if (input.GetType() == typeof(List<double>))//头层
                {
                    for (int i = 0; i < neuron_num; i++)
                    {
                        Neuron neuron = new Neuron(input, name + "(" + i + ")", activation, optimizer);
                        neuron.Father = this;
                        this.Neurons.Add(neuron);
                    }
                }
            }
            else
            {
                for (int i = 0; i < neuron_num; i++)
                {
                    Neuron neuron = new Neuron(null, name + "(" + i + ")", activation, optimizer);
                    neuron.Father = this;
                    this.Neurons.Add(neuron);
                }
            }
        }
        public List<Neuron> Neurons { get; }
        //公有方法
        public new List<double> GetOutput(int isParallel)
        {
          
            
            if (this.Activation == "softmax")
            {
                double[] array = new double[this.Neurons.Count];
                if (isParallel > 0)
                {
                    Parallel.For(0, this.Neurons.Count, i =>
                    {
                        array[i] = this.Neurons[i].GetOutput();
                    });
                }
                else
                {
                    for (int i = 0; i < this.Neurons.Count; i++)
                        array[i] = this.Neurons[i].GetOutput();
                }
                List<double> Output = new List<double>();
                for (int i = 0; i < array.Length; i++)
                {
                    Output.Add(array[i]);
                }
                double sum = Output.Sum(i => Math.Pow(Math.E, i - Output.Max()));
                Output = Output.Select(i => Math.Pow(Math.E, i - Output.Max()) / sum).ToList();
                return Output;
            }
            else
            {
                double[] array = new double[this.Neurons.Count];
                if (isParallel > 0)
                {
                    Parallel.For(0, this.Neurons.Count, i =>
                    {
                        array[i] = this.Neurons[i].GetOutput();
                    });
                }
                else
                {
                    for (int i = 0; i < this.Neurons.Count; i++)
                        array[i] = this.Neurons[i].GetOutput();
                }
                List<double> Output = new List<double>();
                for (int i = 0; i < array.Length; i++)
                {
                    Output.Add(array[i]);
                }
                return Output;
            }
            
        }
        public void LoadNeuron(string FileName)
        {
            var Data = Wineforever.String.client.LoadFromSheet(FileName);
            for (int i = 0; i < this.Neurons.Count; i++)
            {
                this.Neurons[i].Initialization(Data["kernel_0"].Count);
                this.Neurons[i].SetWeight(Data["kernel_" + i.ToString()]);
            }
            this.Father.__isInitialization__ = true;
            this.__kernelPath__ = FileName;
        }
        public new void LoadBias(string FileName)
        {
            var Data = Wineforever.String.client.LoadFromSheet(FileName);
            for (int i = 0; i < this.Neurons.Count; i++)
            {
                this.Neurons[i].SetBias(Data["bias"][i]);
            }
            this.Father.__isInitialization__ = true;
            this.__biasPath__ = FileName;
        }
    }
    //卷积层
    public class ConvLayer : Layer
    {
        //私有字段
        internal string Paddding { get; }
        internal int Step { get; }
        internal int Size { get; }
        internal string __kernelPath__ = "";
        internal string __biasPath__ = "";
        internal int __biasLoadmod__ = 0;
        //公有字段
        public ConvLayer(int kernel_num,int kernel_size, int step = 1, dynamic input = null, string name = null, string padding = "same", string activation = "relu", string optimizer = "GD")
        {
            //---- 创建层 ----
            this.Name = name == null ? "Undefined" : name;
            this.Input = input;
            this.Activation = activation;
            this.Optimizer = optimizer;
            this.Paddding = padding;
            this.Step = step;
            this.Size = kernel_size;
            //---- ----
            //---- 创建卷积核 ----
            this.Kernels = new List<Kernel>();
            //---- ----
            //---- ----
            //判断输入类型
            if (input != null)
            {
                if (input.GetType() == typeof(ConvLayer))//非头层
                {
                    ConvLayer Input = input;
                    for (int i = 0; i < kernel_num; i++)
                    {
                        Kernel kernel = new Kernel(Size, Input, Name + "(" + i + ")", Paddding, Step, Activation, Optimizer);
                        kernel.Father = this;//设置父对象
                        this.Kernels.Add(kernel);
                    }
                }
                else if (input.GetType() == typeof(double[,]))//二维矩阵
                {
                    double[,] Input = input;
                    for (int i = 0; i < kernel_num; i++)
                    {
                        Kernel kernel = new Kernel(Size, Input, Name + "(" + i + ")", Paddding, Step, Activation, Optimizer);
                        kernel.Father = this;//设置父对象
                        this.Kernels.Add(kernel);
                    }
                }
            }
            else
            {
                for (int i = 0; i < kernel_num; i++)
                {
                    Kernel kernel = new Kernel(Size, null, Name + "(" + i + ")", Paddding, Step, Activation, Optimizer);
                    kernel.Father = this;//设置父对象
                    this.Kernels.Add(kernel);
                }
            }
        }
        public List<Kernel> Kernels { get; }
        //公有方法
        public new List<double[,]> GetOutput()
        {
            List<double[,]> Output = new List<double[,]>();
            for (int i = 0; i < Kernels.Count; i++)
            {
                var kernel_output = this.Kernels[i].GetOutput();
                if (kernel_output.GetType() == typeof(double[,]))
                {
                    Output.Add(kernel_output);
                }
                else if (kernel_output.GetType() == typeof(List<double[,]>))
                {
                    Output.Add(Wineforever.Mathematics.client.sum(kernel_output));
                }
            };
            return Output;
        }
        public new void LoadKernel(string FileName,string mod = "channel_last")
        {
            if (this.Input != null)
            {
                if (mod == "channel_first")
                {
                }
                else if (mod == "channel_last")
                {
                    var Data = Wineforever.String.client.Deserialization(FileName);
                    if (this.Input.GetType() == typeof(List<double[,]>))
                    {
                        for (int n = 0; n < this.Kernels.Count; n++)
                        {
                            this.Kernels[n].Initialization(this.Size, Data.GetLength(2));
                            for (int i = 0; i < Data.GetLength(0); i++)
                                for (int j = 0; j < Data.GetLength(1); j++)
                                    for (int k = 0; k < Data.GetLength(2); k++)
                                    {

                                        this.Kernels[n].SetWeight(Data[i, j, k, n],i,j,k);
                                    }
                        }
                    }
                    else if (this.Input.GetType() == typeof(double[,]))
                    {
                        this.Kernels[0].Initialization(this.Size, Data.GetLength(2));
                        for (int i = 0; i < Data.GetLength(0); i++)
                            for (int j = 0; j < Data.GetLength(1); j++)
                                for (int k = 0; k < Data.GetLength(2); k++)
                                {

                                    this.Kernels[0].SetWeight(Data[i, j, k, 1],i,j,k);
                                }
                    }
                    else if ((this.Input.GetType() == typeof(ConvLayer)))
                    {
                        for (int n = 0; n < this.Kernels.Count; n++)
                        {
                            this.Kernels[n].Initialization(this.Size, Input.Kernels.Count);
                            for (int i = 0; i < Size; i++)
                                for (int j = 0; j < Size; j++)
                                    for (int k = 0; k < Input.Kernels.Count; k++)
                                    {

                                        this.Kernels[n].SetWeight(Data[i, j, k, n],i,j,k);
                                    }
                        }
                    }
                    this.Father.__isInitialization__ = true;
                }
            }
            else __kernelPath__ = FileName;
        }
        public new void LoadBias(string FileName,int Format = 0)
        {
            if (this.Input != null)
            {
                if (Format == 0)
                {
                    var Data = Wineforever.String.client.Deserialization(FileName);
                    for (int i = 0; i < Data.GetLength(0); i++)
                    {
                        this.Kernels[i].SetBias(Data[i]);
                    }
                    this.Father.__isInitialization__ = true;
                }
                else if (Format == 1)
                {
                    var Data = Wineforever.String.client.LoadFromSheet(FileName);
                    for (int i = 0; i < this.Kernels.Count(); i++)
                    {
                        this.Kernels[i].SetBias(double.Parse(Data["bias"][i]));
                    }
                    this.Father.__isInitialization__ = true;
                }
            }
            else { __biasPath__ = FileName; __biasLoadmod__ = Format; }
        }
    }
    //平化层
    public class FlattenLayer : Layer
    {
        //公有字段
        public FlattenLayer(dynamic input = null,string name = null)
        {
            //---- 创建层 ----
            this.Name = name == null ? "Flatten" : name;
            this.Input = input;
            //---- ----
        }
        //公有方法
        public new List<double> GetOutput()
        {
            List<double> output = new List<double>();
            //平化操作
            if (this.Input != null)
                if (this.Input.GetType() == typeof(ConvLayer))
                {
                    ConvLayer input = Input;
                    var inputs = input.GetOutput();
                    for (int n = 0; n < inputs.Count; n++)
                        for (int i = 0; i < inputs[n].GetLength(0); i++)
                            for (int j = 0; j < inputs[n].GetLength(1); j++)
                                output.Add(inputs[n][i,j]);
                }
            return output;
        }
    }
    //---- 系统类 ----
    public class System
    {
        //公有字段
        public System(string name = "sys", System input = null)
        {
            //---- 创建系统 ----
            this.Name = name == "" ? "Undefined" : name;
            this.input = input;
            //---- ----
            this.Layers = new List<Layer>();
            this.Update();
        }
        public List<Layer> Layers { get; }//层群
        //私有字段
        internal string Name { get; }//标识名
        internal dynamic input { get; set; }//输入
        internal List<double> errors { get; set; }
        internal double Rand(double min, double max)
        {
            return rand.NextDouble() * (max - min) + min;
        }
        internal bool __isInitialization__ = false;
        internal List<Neuron> __neurons__ = new List<Neuron>();
        internal List<Kernel> __kernels__ = new List<Kernel>();
        internal double __loss__ = 0;//总误差量度
        private Random rand = new Random();
        //超参数
        internal double learning_rate = 0.01;//学习率
        internal double dropout_rate = 0.2;//遗忘概率
        //创建层
        public void Create(int neuron_num, string name = "dense", string activation = "relu", string optimizer = "GD")
        {
            DenseLayer layer;
            if (this.Layers.Count == 0) layer = new DenseLayer(neuron_num, null, name, activation, optimizer);
            else
                layer = new DenseLayer(neuron_num, this.Layers.Last(), name, activation, optimizer);
            Add(layer);
        }
        public void Create(int kernel_num, int kernel_size, int step = 1, string name = "conv", string padding = "same", string activation = "relu", string optimizer = "GD")
        {
            ConvLayer layer;
            if (this.Layers.Count == 0) layer = new ConvLayer(kernel_num, kernel_size, step, null, name, padding, activation, optimizer);
            else
                layer = new ConvLayer(kernel_num, kernel_size, step, this.Layers.Last(), name, padding, activation, optimizer);
            Add(layer);
        }
        public void Create(string name = "flatten")
        {
            FlattenLayer layer;
            if (this.Layers.Count == 0) layer = new FlattenLayer(null, name);
            else
                layer = new FlattenLayer(this.Layers.Last(), name);
            Add(layer);
        }
        //添加层
        public void Add(DenseLayer layer)
        {
            this.Layers.Add(layer);
            layer.Father = this;//设置父对象
            //添加新神经元至群
            for (int i = 0; i < layer.Neurons.Count; i++)
                __neurons__.Add(layer.Neurons[i]);
            Update();
        }
        public void Add(ConvLayer layer)
        {
            this.Layers.Add(layer);
            layer.Father = this;//设置父对象
            //添加新卷积核至群
            for (int i = 0; i < layer.Kernels.Count; i++)
            {
                __kernels__.Add(layer.Kernels[i]);
                layer.Kernels[i].Input = Layers[Math.Max(Layers.Count - 2,0)];
            }
            Update();
        }
        public void Add(FlattenLayer layer)
        {
            this.Layers.Add(layer);
            layer.Father = this;//设置父对象
            Update();
        }
        //删除层
        public void Remove(Layer layer)
        {
            this.Layers.Remove(layer);
            layer.Father = null;//撤销关系
            if (this.Layers.Count > 0) Update();
        }
        //训练
        public void Train(List<double> target, int epoch, string loss = "MBE")
        {
            __isInitialization__ = true;//开始训练之后，停止变动初始权重
            if (this.Layers.Last().GetType() == typeof(DenseLayer))
            {
                DenseLayer layer = this.Layers.Last() as DenseLayer;
                for (int T = 0; T < epoch; T++)
                {
                    __loss__ = 0;//初始化代价值
                    List<double> Outputs = new List<double>();
                    this.errors = new List<double>();
                    for (int i = 0; i < layer.Neurons.Count; i++)
                    {
                        Outputs.Add(layer.Neurons[i].GetOutput());
                        if (loss == "MBE")//平均偏差误差
                        {
                            loss += Outputs[i] - target[i];
                            errors.Add(Outputs[i] - target[i]);
                        }
                        else if (loss == "MAE")//平均绝对误差
                        {
                            loss += Math.Abs(Outputs[i] - target[i]);
                            errors.Add(Math.Abs(Outputs[i] - target[i]));
                        }
                        else if (loss == "MSE")//均方误差
                        {
                            loss += 0.5 * Math.Pow(Outputs[i] - target[i], 2.0);
                            errors.Add(0.5 * Math.Pow(Outputs[i] - target[i], 2.0));
                        }
                        else if (loss == "SVM")//分类损失
                        {
                            loss += Math.Max(0, Outputs[i] - target[i] + 1);
                            errors.Add(Math.Max(0, Outputs[i] - target[i] + 1));
                        }
                        else if (loss == "crossentropy")//交叉熵
                        {
                            loss += -(Outputs[i] * Math.Log(target[i])) + (1 - Outputs[i]) * Math.Log(1 - target[i]);
                            errors.Add(-(Outputs[i] * Math.Log(target[i])) + (1 - Outputs[i]) * Math.Log(1 - target[i]));
                        }
                        else//调用外部脚本
                        {
                            //---- 自定义损失函数 ----
                            string location = AppDomain.CurrentDomain.BaseDirectory + "loss\\" + loss + ".lua";
                            Lua lua = new Lua();
                            lua.DoString("y_real=" + Outputs[i]);
                            lua.DoString("y_pre=" + target[i]);
                            lua.DoString(Wineforever.String.client.Load(location));
                            var res = (double)lua.DoString("return loss")[0];
                            loss += res;
                            errors.Add(res);
                        }
                    }
                    //将每一项误差传递给最后一层的每个神经元处理
                    for (int i = 0; i < layer.Neurons.Count; i++)
                    {
                        layer.Neurons[i].Update(errors[i], layer.Neurons[i].Optimizer);
                    }
                    Console.WriteLine("[{0}] epoch:{1},loss:{2}", this.Name, T, __loss__.ToString(".000"));
                }
            }
        }
        //输出
        public dynamic GetOutput(int isParallel = 1)
        {
            if (Layers.Last().GetType() == typeof(DenseLayer))
            {
                return (Layers.Last() as DenseLayer).GetOutput(isParallel);
            }
            else if (Layers.Last().GetType() == typeof(ConvLayer))
            {
                return (Layers.Last() as ConvLayer).GetOutput();
            }
            else if (Layers.Last().GetType() == typeof(FlattenLayer))
            {
                return (Layers.Last() as FlattenLayer).GetOutput();
            }
            return -1;
        }
        //设置系统输入
        public void Input(System system)
        {
            this.input = system;
            this.Layers.First().Input = system;
            Update();
        }
        public void Input(List<double> samples)
        {
            this.input = samples;
            this.Layers.First().Input = samples;
            Update();
        }
        public void Input(List<double[,]> samples)
        {
            this.input = samples;
            this.Layers.First().Input = samples;
            Update();
        }
        //私有方法
        internal void Update()//更新连接
        {
            if (input != null)
            {
                //如果存在平化层，对其后层进行初始化
                for (int i = 0; i < Layers.Count; i++)
                {
                    if (Layers[i].GetType() == typeof(FlattenLayer) && Layers[Math.Min(i + 1, Layers.Count - 1)].GetType() == typeof(DenseLayer))
                    {
                        int num = 0;
                        DenseLayer denseLayer = (Layers[Math.Min(i + 1, Layers.Count - 1)] as DenseLayer);
                            Layer preLayer = Layers[Math.Max(i - 1, 0)];
                            if (preLayer.GetType() == typeof(ConvLayer))
                            {
                                num = 1;
                                num *= (preLayer as ConvLayer).Kernels.Count;
                            }
                            else if (preLayer.GetType() == typeof(DenseLayer))
                            {
                                num = 1;
                                num *= (preLayer as DenseLayer).Neurons.Count;
                            }
                            if (input.GetType() == typeof(List<double>))
                            {
                                num *= (input as List<double>).Count;
                            }
                            else if (input.GetType() == typeof(double[,]))
                            {
                                num *= (input as double[,]).GetLength(0) * (input as double[,]).GetLength(1);
                            }
                            else if (input.GetType() == typeof(List<double[,]>))
                            {
                                num *= (input as List<double[,]>)[0].GetLength(0) * (input as List<double[,]>)[0].GetLength(1);
                            }
                            for (int n = 0; n < denseLayer.Neurons.Count; n++)
                                denseLayer.Neurons[n].Initialization(num);
                        denseLayer.LoadKernel(denseLayer.__kernelPath__);
                        denseLayer.LoadBias(denseLayer.__biasPath__);
                        break;
                    }
                }
                if (input.GetType() == typeof(System))
                {
                    if (Layers.First().GetType() == typeof(DenseLayer))
                    {
                        var layer = Layers.First() as DenseLayer;
                        for (int i = 0; i < layer.Neurons.Count; i++)
                        {
                            Neuron neuron = layer.Neurons[i];
                            neuron.Input = this.input.Layers[this.input.Layers.Count - 1].Neurons;
                            if (neuron.Weights.Count <= 0) neuron.Initialization(this.input.Layers[this.input.Layers.Count - 1].Neurons.Count);
                        }
                    }
                }
                else if (input.GetType() == typeof(List<double>))
                {
                    if (Layers.First().GetType() == typeof(DenseLayer))
                    {
                        var layer = Layers.First() as DenseLayer;
                        for (int i = 0; i < layer.Neurons.Count; i++)
                        {
                            Neuron neuron = layer.Neurons[i];
                            neuron.Input = input;
                            if (neuron.Weights.Count <= 0) neuron.Initialization(input.Count);
                        }
                    }
                }
                else if (input.GetType() == typeof(double[,]))
                {
                    if (Layers.First().GetType() == typeof(ConvLayer))
                    {
                        var layer = Layers.First() as ConvLayer;
                        for (int i = 0; i < layer.Kernels.Count; i++)
                        {
                            Kernel kernel = layer.Kernels[i];
                            kernel.Input = input;
                        }
                    }
                }
                else if (input.GetType() == typeof(List<double[,]>))
                {
                    if (Layers.First().GetType() == typeof(ConvLayer))
                    {
                        var layer = Layers.First() as ConvLayer;
                        for (int i = 0; i < layer.Kernels.Count; i++)
                        {
                            Kernel kernel = layer.Kernels[i];
                            kernel.Input = input;
                        }
                    }
                }
            }
            //初始化权重
            if (!__isInitialization__)
            {
                for (int i = 0; i < __neurons__.Count; i++)
                {
                    __neurons__[i].SetWeight(1);
                    __neurons__[i].SetBias(0);
                }
                for (int i = 0; i <__kernels__.Count; i++)
                {
                    __kernels__[i].SetWeight(1);
                    __kernels__[i].SetBias(0);
                }
                if (Layers.Count>0 && Layers.First().Input != null && Layers.First().GetType() == typeof(ConvLayer))
                {
                    ConvLayer layer = (Layers.First() as ConvLayer);
                    Type input_type = layer.Input.GetType();
                    for (int i = 0; i < layer.Kernels.Count(); i++)
                    {
                        if (input_type == typeof(double[,]))
                        {
                            layer.Kernels[i].Initialization(layer.Size, 1);
                        }
                        else if (input_type == typeof(List<double[,]>))
                        {
                            layer.Kernels[i].Initialization(layer.Size, layer.Input.Count);
                        }
                        else if (input_type == typeof(ConvLayer))
                        {
                            layer.Kernels[i].Initialization(layer.Size, layer.Input.Kernels.Count());
                        }
                    }
                }
            }
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].GetType() == typeof(ConvLayer) && Layers[i].Input != null && (Layers[i] as ConvLayer).__kernelPath__ != "")
                {
                    Layers[i].LoadKernel((Layers[i] as ConvLayer).__kernelPath__);
                }
                if (Layers[i].GetType() == typeof(ConvLayer) && Layers[i].Input != null && (Layers[i] as ConvLayer).__biasPath__ != "")
                {
                    Layers[i].LoadBias((Layers[i] as ConvLayer).__biasPath__, (Layers[i] as ConvLayer).__biasLoadmod__);
                }
            }
        }
        //打印
        public void Print()
        {
            Console.WriteLine(" -- System Name:{0}",this.Name);
            for (int i = 0; i < this.Layers.Count; i++)
            {
                Console.WriteLine(" ------ Layer Name:{0}",this.Layers[i].Name);
                var layer = Layers[i];
                if (layer.GetType() == typeof(DenseLayer))
                {
                    for (int j = 0; j < (layer as DenseLayer).Neurons.Count; j++)
                    {
                        Console.WriteLine(" ---------- Neuron Name:{0}", (layer as DenseLayer).Neurons[j].Name);
                    }
                }
                else if (layer.GetType() == typeof(ConvLayer))
                {
                    for (int j = 0; j < (layer as ConvLayer).Kernels.Count; j++)
                    {
                        Console.WriteLine(" ---------- Kernel Name:{0}", (layer as ConvLayer).Kernels[j].Name);
                    }
                }
            }
        }
        //保存
        public void Save()
        {
            for (int i = 0; i < __neurons__.Count; i++)
            {
                __neurons__[i].Save();
            }
        }
        //读取
        public void Load(string Path)
        {
            int conv_num = 1;
            int dense_num = 1;
            for (int i = 0; i < Layers.Count; i++)
            {
                if (Layers[i].GetType() == typeof(ConvLayer))
                {
                    var layer = Layers[i];
                    layer.LoadKernel(Path + "\\conv_" + conv_num.ToString() + "_kernel.group");
                    layer.LoadBias(Path + "\\conv_" + conv_num.ToString() + "_bias.wf", 1);
                    conv_num++;
                }
                else if (Layers[i].GetType() == typeof(DenseLayer))
                {
                    var layer = Layers[i];
                    layer.LoadKernel(Path + "\\dense_" + dense_num.ToString() + "_kernel.wf");
                    layer.LoadBias(Path + "\\dense_" + dense_num.ToString() + "_bias.wf", 1);
                    dense_num++;
                }
            } 
        }
    }
}
