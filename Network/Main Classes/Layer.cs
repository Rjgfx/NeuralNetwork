using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Network
{
    class Layer
    {
        public List<Neuron> neurons;
        public int number_of_neurons;
        public double[] input;
        public Func<double, double> ActivationFunction;

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

        public void SetLocalGradient(double[] errors, Func<double, double> ActivationFunctionDerivative)
        {
            for (int i = 0; i < number_of_neurons; i++)
            {
                neurons[i].LocalGradient = errors[i] * ActivationFunctionDerivative(neurons[i].InducedLocalField);
            }
        }

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

        public void СhangeTheWeightsAndBias(double learningrate)
        {
            for (int i = 0; i < number_of_neurons; i++)
            {
                neurons[i].СhangeTheWeightsAndBias(input, learningrate);
            }

        }
    }
}
