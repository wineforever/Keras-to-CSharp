using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Diagnostics;
using System.Linq.Expressions;

namespace mnist
{
    public partial class mainfrm : Form
    {
        public mainfrm()
        {
            InitializeComponent();
        }
        public Wineforever.Neuralnet.System sys;
        private void mainfrm_Load(object sender, EventArgs e)
        {
            //初始化手写操作
            handwriting_ini();
            //构建卷积神经网络
            sys = new Wineforever.Neuralnet.System("sys");
            sys.Create(3, 5, 1, "卷积层 1"); //3个卷积核，5x5尺寸
            sys.Create(3, 3, 1, "卷积层 2"); //3个卷积核，3x3尺寸
            sys.Create("Flatten层");
            sys.Create(128, "全连接层 1"); //128个神经元
            sys.Create(10, "全连接层 2", "softmax"); //10个神经元，Softmax输出
            //加载权重
            sys.Load(System.AppDomain.CurrentDomain.BaseDirectory + "Model\\");
        }
        private void predictBtn_Click(object sender, EventArgs e)
        {
            //耗时计算
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            //预处理
            bitmap = new Bitmap(bitmap, 28, 28);
            BitmapData data = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
            var input = new List<float[,]>();
            var map = new float[28, 28];
            unsafe//指针取色
            {
                for (int i = 0; i < bitmap.Width; i++)
                    for (int j = 0; j < bitmap.Height; j++)
                    {
                        byte* ptr = (byte*)data.Scan0 + j * 3 + i * data.Stride;
                        map[i, j] = *ptr / 255;
                    }
            }
            bitmap.UnlockBits(data);
            input.Add(map);
            //载入数据
            sys.Input(input);
            //预测
            List<float> output = sys.GetOutput();
            var sorted = output.Select((x, i) => new KeyValuePair<float, int>(x, i)).OrderByDescending(x => x.Key).ToList();
            var index = sorted.Select(i => i.Value).ToList()[0];
            res.Text = index.ToString();
            //打印
            stopwatch.Stop();
            Console.WriteLine("总用时(毫秒)：" + stopwatch.ElapsedMilliseconds);
        }
        private void exitBtn_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }
        #region 写字过程
        private Bitmap bitmap;
        private Graphics g;
        //初始化
        private void handwriting_ini()
        {
            bitmap = new Bitmap(inputbox.Width, inputbox.Height);
            g = Graphics.FromImage(bitmap);
            g.FillRectangle(Brushes.Black, 0, 0, inputbox.Width, inputbox.Height);
            inputbox.Image = bitmap;
        }
        //右键清除绘图痕迹
        private void inputbox_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
            {
                handwriting_ini();
            }
        }
        //开始写字
        private void inputbox_MouseDown(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
                timer.Enabled = true;
        }
        //结束写字
        private void inputbox_MouseUp(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left)
                timer.Enabled = false;
        }
        //写字过程
        private void timer_Tick(object sender, EventArgs e)
        {
            g.FillEllipse(Brushes.White, inputbox.PointToClient(Control.MousePosition).X, inputbox.PointToClient(Control.MousePosition).Y, 20, 20);
            inputbox.Image = bitmap;
        }
        #endregion
    }
}
