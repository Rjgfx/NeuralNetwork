using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Network
{
    public static class ActivationFunctions
    {

        private const double a = 1.0;


        public static double ThresholdFunction(double x)
        {
            return x >= 0.0 ? 1.0 : 0.0;
        }

        public static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double dsigmoid(double x)
        {
            double factor = a * Math.Pow(Math.E, -a * x);

            return factor * Math.Pow(sigmoid(x), 2.0);
        }

        // Функція гіперболічного тангенсу
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        // Похідна функції гіперболічного тангенсу
        public static double TanhDerivative(double x)
        {
            double tanh = Tanh(x);
            return 1.0 - tanh * tanh;
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1 : 0;
        }

        // Функція PReLU (Parametric Rectified Linear Unit)
        public static double PReLU(double x, double alpha)
        {
            return x >= 0 ? x : alpha * x;
        }

        // Похідна функції PReLU
        public static double PReLUDerivative(double x, double alpha)
        {
            return x >= 0 ? 1 : alpha;
        }

        // Функція RReLU (Randomized Leaky Rectified Linear Unit)
        public static double RReLU(double x)
        {
            double lower = 1;
            double upper = 0;
            if (x >= 0)
                return x;
            else
                return new Random().NextDouble() * (upper - lower) + lower;
        }

        // Похідна функції RReLU
        public static double RReLUDerivative(double x)
        {
            double lower = 1;
            double upper = 0;
            return x >= 0 ? 1 : (lower + upper) / 2.0;
        }
    }
}
