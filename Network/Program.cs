using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Network
{
    internal static class Program
    {
        [STAThread]

        static void Main()  
        {

            int[] sizes_of_layers = new int[] { 784, 16, 16, 10 };
            NeuralNetwork nn = new NeuralNetwork(sizes_of_layers, ActivationFunctions.sigmoid, ActivationFunctions.dsigmoid, 
                ActivationFunctions.sigmoid, ActivationFunctions.dsigmoid, 0.1);

            nn.Train();
            nn.Test();

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
        }
    }
}


