using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Network
{
    class Neuron
    {
        /// <summary>
        /// Індуковане локальне поле
        /// </summary>
        public double InducedLocalField { get; private set; }
        /// <summary>
        /// Локальний градієнт
        /// </summary>
        public double LocalGradient { get; set; }
        /// <summary>
        /// Вагові коефіцієнти
        /// </summary>
        public double[] weights { get; set; }
        /// <summary>
        /// Зміщення
        /// </summary>
        public double bias { get; set; }
        /// <summary>
        /// Функція активації
        /// </summary>
        public Func<double, double> ActivationFunction { get; private set; }


        /// <summary>
        /// Створює нерон
        /// </summary>
        /// <param name="size">розмір попереднього шару</param>
        /// <param name="ActivationFunction">функція активації</param>
        public Neuron(int size, Func<double, double> ActivationFunction)
        {
            this.ActivationFunction = ActivationFunction;
            weights = new double[size];
        }

        /// <summary>
        /// Встановлює індуковане локальне поле
        /// </summary>
        /// <param name="input">вхідний сигнал</param>
        public void Feed(double[] input)
        {
            InducedLocalField = 0;
            for (int i = 0; i < input.Length; i++)
            {
                InducedLocalField += weights[i] * input[i];
            }
            InducedLocalField += bias;

        }

        /// <summary>
        /// Корегує вагові коефіцієнти
        /// </summary>
        /// <param name="input">вхідний сигнал</param>
        /// <param name="learningrate">швидкість навчання</param>
        public void СhangeTheWeightsAndBias(double[] input, double learningrate)
        {

            bias += learningrate * LocalGradient;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += learningrate * LocalGradient * input[i];
            }
        }
    }
}
