using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Network
{
    public static class ActivationFunctions
    {
        /// <summary>
        /// Параметр для сигмоїдальної функції
        /// </summary>
        private const double a = 1.0;

        /// <summary>
        /// Сигмоїдальна функція 
        /// </summary>
        /// <param name="x">аргумент функції</param>
        /// <returns></returns>
        public static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        /// <summary>
        /// Похідна сигмоїдальної функціі
        /// </summary>
        /// <param name="x">аргумент функції</param>
        /// <returns></returns>
        public static double dsigmoid(double x)
        {
            double factor = a * Math.Pow(Math.E, -a * x);

            return factor * Math.Pow(sigmoid(x), 2.0);
        }

        /// <summary>
        /// Функція гіперболічного тангенсу
        /// </summary>
        /// <param name="x">аргумент функції</param>
        /// <returns></returns>
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        /// <summary>
        /// Похідна функціх гіперболічного тангенсу
        /// </summary>
        /// <param name="x">аргумент функції</param>
        /// <returns></returns>
        public static double TanhDerivative(double x)
        {
            double tanh = Tanh(x);
            return 1.0 - tanh * tanh;
        }
    }
}
