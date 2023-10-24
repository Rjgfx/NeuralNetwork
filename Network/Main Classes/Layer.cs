using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Network
{
    class Layer
    {
        /// <summary>
        /// Нейрони
        /// </summary>
        public List<Neuron> neurons;
        /// <summary>
        /// Кількість неронів
        /// </summary>
        public int number_of_neurons;
        /// <summary>
        /// Вхідний сигнал
        /// </summary>
        public double[] input;
        /// <summary>
        /// Функція активації
        /// </summary>
        public Func<double, double> ActivationFunction;

        /// <summary>
        /// Ініціалізує шар за параметрами
        /// </summary>
        /// <param name="size_of_previous_layer">кількість нейронів у попередньому шарі</param>
        /// <param name="number_of_neurons">кількість неронів у шарі</param>
        /// <param name="ActivationFunction">функція активації</param>
        public Layer(int size_of_previous_layer, int number_of_neurons, Func<double, double> ActivationFunction)
        {
            this.number_of_neurons = number_of_neurons;
            this.ActivationFunction = ActivationFunction;
            neurons = new List<Neuron>();

            for (int i = 0; i < number_of_neurons; i++)
            {
                neurons.Add(new Neuron(size_of_previous_layer, ActivationFunction));

            }
            Random random = new Random();

            for (int i = 0; i < number_of_neurons; i++)
            {

                neurons[i].bias = random.NextDouble() * 2 - 1;
                for (int j = 0; j < neurons[i].weights.Length; j++)
                {
                    neurons[i].weights[j] = random.NextDouble() * 2 - 1;
                }

            }
        }

        /// <summary>
        /// Встановлює локальні градієнти вихідного шару
        /// </summary>
        /// <param name="errors">відхилення від цілі</param>
        /// <param name="ActivationFunctionDerivative">похідна функції активації</param>
        public void SetLocalGradient(double[] errors, Func<double, double> ActivationFunctionDerivative)
        {
            for (int i = 0; i < number_of_neurons; i++)
            {
                neurons[i].LocalGradient = errors[i] * ActivationFunctionDerivative(neurons[i].InducedLocalField);
            }
        }

        /// <summary>
        /// Встановлює локальні градієнти прихованих шарів
        /// </summary>
        /// <param name="next_layer">відхилення від цілі</param>
        /// <param name="ActivationFunctionDerivative">похідна функції активації</param>
        public void SetLocalGradient(Layer next_layer, Func<double, double> ActivationFunctionDerivative)
        {
            double inner_sum = 0;
            for (int i = 0; i < number_of_neurons; ++i)
            {
                for (int j = 0; j < next_layer.number_of_neurons; j++)
                {
                    inner_sum += next_layer.neurons[j].weights[i] * next_layer.neurons[j].LocalGradient;
                }
                neurons[i].LocalGradient = ActivationFunctionDerivative(neurons[i].InducedLocalField) * inner_sum;
            }
        }

        /// <summary>
        /// Корегує вагові коефіціжнти нейронів цього шару
        /// </summary>
        /// <param name="learningrate">швидкість навчання</param>
        public void СhangeTheWeightsAndBias(double learningrate)
        {
            for (int i = 0; i < number_of_neurons; i++)
            {
                neurons[i].СhangeTheWeightsAndBias(input, learningrate);
            }

        }
    }
}
