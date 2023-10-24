using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Network
{
    class Neuron
    {
        public double InducedLocalField { get; private set; }
        public double LocalGradient { get; set; }   
        public double[] weights { get; set; }
        public double bias { get; set; }
        public double output { get; set; }
        public Func<double, double> ActivationFunction { get; private set; }    

        public Neuron(int size, Func<double, double> ActivationFunction)
        {
            this.ActivationFunction = ActivationFunction;
            weights = new double[size];
        }

        public void Feed(double[] input)
        {
            InducedLocalField = 0;
            for (int i = 0; i < input.Length; i++)
            {
                InducedLocalField += weights[i] * input[i];
            }
            InducedLocalField += bias;

        }

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
