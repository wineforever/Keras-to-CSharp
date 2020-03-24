namespace mnist
{
    partial class mainfrm
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.inputbox = new System.Windows.Forms.PictureBox();
            this.res = new System.Windows.Forms.Label();
            this.predictBtn = new System.Windows.Forms.Button();
            this.exitBtn = new System.Windows.Forms.Button();
            this.timer = new System.Windows.Forms.Timer(this.components);
            ((System.ComponentModel.ISupportInitialize)(this.inputbox)).BeginInit();
            this.SuspendLayout();
            // 
            // inputbox
            // 
            this.inputbox.Location = new System.Drawing.Point(12, 12);
            this.inputbox.Name = "inputbox";
            this.inputbox.Size = new System.Drawing.Size(280, 280);
            this.inputbox.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.inputbox.TabIndex = 0;
            this.inputbox.TabStop = false;
            this.inputbox.MouseClick += new System.Windows.Forms.MouseEventHandler(this.inputbox_MouseClick);
            this.inputbox.MouseDown += new System.Windows.Forms.MouseEventHandler(this.inputbox_MouseDown);
            this.inputbox.MouseUp += new System.Windows.Forms.MouseEventHandler(this.inputbox_MouseUp);
            // 
            // res
            // 
            this.res.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.res.Font = new System.Drawing.Font("华文宋体", 99.74999F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(134)));
            this.res.Location = new System.Drawing.Point(298, 13);
            this.res.Name = "res";
            this.res.Size = new System.Drawing.Size(280, 280);
            this.res.TabIndex = 1;
            this.res.Text = "0";
            this.res.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // predictBtn
            // 
            this.predictBtn.Location = new System.Drawing.Point(584, 13);
            this.predictBtn.Name = "predictBtn";
            this.predictBtn.Size = new System.Drawing.Size(91, 29);
            this.predictBtn.TabIndex = 2;
            this.predictBtn.Text = "&Predict";
            this.predictBtn.UseVisualStyleBackColor = true;
            this.predictBtn.Click += new System.EventHandler(this.predictBtn_Click);
            // 
            // exitBtn
            // 
            this.exitBtn.Location = new System.Drawing.Point(584, 48);
            this.exitBtn.Name = "exitBtn";
            this.exitBtn.Size = new System.Drawing.Size(91, 29);
            this.exitBtn.TabIndex = 3;
            this.exitBtn.Text = "&Exit";
            this.exitBtn.UseVisualStyleBackColor = true;
            this.exitBtn.Click += new System.EventHandler(this.exitBtn_Click);
            // 
            // timer
            // 
            this.timer.Interval = 1;
            this.timer.Tick += new System.EventHandler(this.timer_Tick);
            // 
            // mainfrm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(682, 302);
            this.Controls.Add(this.exitBtn);
            this.Controls.Add(this.predictBtn);
            this.Controls.Add(this.res);
            this.Controls.Add(this.inputbox);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Name = "mainfrm";
            this.Text = "mnist ocr";
            this.Load += new System.EventHandler(this.mainfrm_Load);
            ((System.ComponentModel.ISupportInitialize)(this.inputbox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox inputbox;
        private System.Windows.Forms.Label res;
        private System.Windows.Forms.Button predictBtn;
        private System.Windows.Forms.Button exitBtn;
        private System.Windows.Forms.Timer timer;
    }
}

