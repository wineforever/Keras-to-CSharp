using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Wineforever.Mathematics
{
    public class client
    {
        public static double[,] para_conv(double[,] input, double[,] kernel, double bias = 0, int step = 1, string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,] input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            Parallel.For(0, output_width, i =>
            {
                Parallel.For(0, output_height, j =>
                {
                    int X = i * step;
                    int Y = j * step;
                    double sum = 0;
                    for (int m = 0; m < kernel_width; m++)
                        for (int n = 0; n < kernel_height; n++)
                            sum += input_padding[X + m, Y + n] * kernel[m,n];
                    output[i, j] = sum + bias;
                });
            });
            return output;
        }
        public static double[,] conv(double[,] input, double[,] kernel, double bias = 0, int step = 1, string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,] input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    int X = i * step;
                    int Y = j * step;
                    double sum = 0;
                    for (int m = 0; m < kernel_width; m++)
                        for (int n = 0; n < kernel_height; n++)
                            sum += input_padding[X + m, Y + n] * kernel[m,n];
                    output[i, j] = sum + bias;
                }
            return output;
        }
        public static double[,] conv(double[,] input, double[,] kernel, double bias = 0, int step = 1, string activator = "relu", string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            double[,] input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    int X = i * step;
                    int Y = j * step;
                    double sum = 0;
                    for (int m = 0; m < kernel_width; m++)
                        for (int n = 0; n < kernel_height; n++)
                            sum += input_padding[X + m, Y + n] * kernel[m,n];
                    output[i, j] = activation(sum + bias, activator);
                }
            return output;
        }
        public static double[,] conv(List<double[,]> input, double[,,] kernel, double bias = 0, int step = 1, string activator = "relu", string padding_mod = "same")
        {
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int kernel_channel = kernel.GetLength(2);
            int width = input[0].GetLength(0);
            int height = input[0].GetLength(1);
            int input_channel = input.Count();
            List<double[,]> input_padding = null;
            int output_width = 0;
            int output_height = 0;
            if (padding_mod == "same")
            {
                output_width = (int)System.Math.Ceiling((double)width / (double)step);
                output_height = (int)System.Math.Ceiling((double)height / (double)step);
                input_padding = padding(input, kernel, step, padding_mod);
            }
            double[,] output = new double[output_width, output_height];
            var res = new double[output_width, output_height];
            for (int i = 0; i < output_width; i++)
                for (int j = 0; j < output_height; j++)
                {
                    double channel_sum = 0;
                    //Input Channel == Kernel Channel
                    for (int u = 0; u < input_channel; u++)
                    {
                        int X = i * step;
                        int Y = j * step;
                        for (int m = 0; m < kernel_width; m++)
                            for (int n = 0; n < kernel_height; n++)
                                channel_sum += input_padding[u][X + m, Y + n] * kernel[m, n, u];
                      
                    }
                    channel_sum += bias;
                    //normalize like Keras!
                    //do normalization first then add activation.
                    output[i, j] = activation(channel_sum, activator);
                }
            return output;
        }
        public static double[,] padding(double[,] input, double[,] kernel, int step = 1, string padding_mod = "same")
        {
            double[,] output = null;
            int width = input.GetLength(0);
            int height = input.GetLength(1);
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            if (padding_mod == "same")
            {
                int _width = (int)System.Math.Ceiling((double)width / (double)step);
                int _height = (int)System.Math.Ceiling((double)height / (double)step);
                int pad_needed_width = (_width - 1) * step + kernel_width - width;
                int pad_left = pad_needed_width / 2;
                int pad_right = pad_needed_width - pad_left;
                int pad_needed_height = (_height - 1) * step + kernel_height - height;
                int pad_top = pad_needed_height / 2;
                int pad_bottom = pad_needed_height - pad_top;
                output = new double[pad_left + width + pad_right, pad_top + height + pad_bottom];
                for (int i = pad_left; i < width + pad_left; i++)
                    for (int j = pad_bottom; j < height + pad_bottom; j++)
                        output[i, j] = input[i - pad_left, j - pad_bottom];
                return output;
            }
            return input;
        }
        public static List<double[,]> padding(List<double[,]> input, double[,,] kernel, int step = 1, string padding_mod = "same")
        {
            List<double[,]> output = new List<double[,]>(); ;
            int width = input[0].GetLength(0);
            int height = input[0].GetLength(1);
            int input_channel = input.Count;
            int kernel_width = kernel.GetLength(0);
            int kernel_height = kernel.GetLength(1);
            int kernel_channel = kernel.GetLength(2);
            if (padding_mod == "same")
            {
                int _width = (int)System.Math.Ceiling((double)width / (double)step);
                int _height = (int)System.Math.Ceiling((double)height / (double)step);
                int pad_needed_width = (_width - 1) * step + kernel_width - width;
                int pad_left = pad_needed_width / 2;
                int pad_right = pad_needed_width - pad_left;
                int pad_needed_height = (_height - 1) * step + kernel_height - height;
                int pad_top = pad_needed_height / 2;
                int pad_bottom = pad_needed_height - pad_top;
                //FUCK THE BUG!
                //var res = new double[pad_left + width + pad_right, pad_top + height + pad_bottom];
                for (int n = 0; n < input_channel; n++)
                {
                    var res = new double[pad_left + width + pad_right, pad_top + height + pad_bottom];
                    for (int i = pad_left; i < width + pad_left; i++)
                        for (int j = pad_bottom; j < height + pad_bottom; j++)
                            res[i, j] = input[n][i - pad_left, j - pad_bottom];
                    output.Add(res);
                }
            }
            return output;
        }
        public static double[,] sum(List<double[,]> inputs)
        {
            var width = inputs.Count > 0 ? inputs[0].GetLength(0) : 0;
            var height = inputs.Count > 0 ? inputs[0].GetLength(1) : 0;
            double[,] output = new double[width, height];
            for (int n = 0; n < inputs.Count; n++)
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++)
                        output[i, j] += inputs[n][i, j];
            return output;
        }
        public static double activation(double x, string func)
        {
            if (func == "relu") return x > 0 ? x : 0;
            else if (func == "normal") return x;
            else if (func == "sigmoid") return 1.0 / (1.0 + Math.Pow(Math.E, -x));
            else if (func == "leaky relu") return x > 0 ? x : 0.01 * x;
            else return 0;
        }
        public static void print(double[,] input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append(input[i, j] + " ");
                }
                sb.Append("\r\n");
            }
            Console.Write(sb.ToString());
        }
        public static void print(double[,,] input)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < input.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append("[");
                    for (int k=0;k<input.GetLength(2);k++)
                    sb.Append(input[i, j,k] + " ");
                    sb.Append("]");
                }
                sb.Append("]");
            }
            sb.Append("]\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(double[,,,] input)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");
            for (int i = 0; i < input.GetLength(0); i++)
            {
                sb.Append("[");
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    sb.Append("[");
                    for (int k = 0; k < input.GetLength(2); k++)
                    {
                        sb.Append("[");
                        for (int l = 0; l < input.GetLength(3); l++)
                            sb.Append(input[i, j, k, l] + " ");
                        sb.Append("]");
                    }
                    sb.Append("]");
                }
                sb.Append("]");
            }
            sb.Append("]\r\n");
            Console.Write(sb.ToString());
        }
        public static void print(List<double[,]> input)
        {
            StringBuilder sb = new StringBuilder();
            for (int n = 0; n < input.Count; n++)
            {
                sb.Append("\r\n---- Depth:" + n + " ----\r\n");
                for (int i = 0; i < input[n].GetLength(0); i++)
                {
                    for (int j = 0; j < input[n].GetLength(1); j++)
                    {
                        sb.Append(input[n][i, j] + " ");
                    }
                    sb.Append("\r\n");
                }
                sb.Append("---- ------- ----\r\n");
            }
            Console.Write(sb.ToString());
        }
        public static void print(List<double> input)
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < input.Count; i++)
            {
                sb.Append(input[i] + "  ");
                if ((i + 1) % 10 == 0) sb.Append("\r\n");
            }
            sb.Append("\r\n");
            Console.Write(sb.ToString());
        }
        public static int[] get_shape(double[,] input)
        {
            int[] res = new int[2];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            Console.WriteLine("(" + res[0] + "," + res[1] + ")");
            return res;
        }
        public static int[] get_shape(double[,,] input)
        {
            int[] res = new int[3];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            res[2] = input.GetLength(2);
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + ")");
            return res;
        }
        public static int[] get_shape(double[,,,] input)
        {
            int[] res = new int[4];
            res[0] = input.GetLength(0);
            res[1] = input.GetLength(1);
            res[2] = input.GetLength(2);
            res[3] = input.GetLength(3);
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + "," + res[3] + ")");
            return res;
        }
        public static int[] get_shape(List<double[,]> input)
        {
            int[] res = new int[3];
            res[0] = input.Count;
            if (input.Count > 0)
                res[1] = input[0].GetLength(0);
            else res[1] = 0;
            if (input.Count > 0)
                res[2] = input[0].GetLength(1);
            else res[2] = 0;
            Console.WriteLine("(" + res[0] + "," + res[1] + "," + res[2] + ")");
            return res;
        }
        public static int[] get_shape(List<double> input)
        {
            int[] res = new int[1];
            res[0] = input.Count;
            Console.WriteLine("(" + res[0] + ")");
            return res;
        }
    }
}
