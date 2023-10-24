using System;
using System.IO;
using System.CodeDom.Compiler;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Globalization;


namespace Network
{
    class NeuralNetwork
    {
        /// <summary>
        /// Кількість прихованих шарів
        /// </summary>
        private int number_of_hidden_layers;
        /// <summary>
        /// Приховані шари
        /// </summary>
        private List<Layer> hiddenlayers = new List<Layer>();
        /// <summary>
        /// Вихідний шар
        /// </summary>
        private Layer outputlayer;
        /// <summary>
        /// Швидкість навчання
        /// </summary>
        private double learningrate;
        /// <summary>
        /// Похідна функції активації прихованих шарів
        /// </summary>
        private Func<double, double> ActivationFunctionDerivative;
        /// <summary>
        /// Похідна функції активації вихідного шару
        /// </summary>
        private Func<double, double> ActivationFunctionDerivativeOutputLayers;

        /// <summary>
        /// Ініціалізує неромережу за допомогою параметрів
        /// </summary>
        /// <param name="sizes_of_layers">кількість нейронів у кожному з шарів</param>
        /// <param name="ActivationFunction">функція активації прихованих шарів</param>
        /// <param name="ActivationFunctionDerivative">похідна функції активації прихованих шарі</param>
        /// <param name="ActivationFunctionOutputLayers">функція активації вихідного шару</param>
        /// <param name="ActivationFunctionDerivativeOutputLayers">похідна функції активації вихідного шау</param>
        /// <param name="learningrate">швидкість навчання</param>
        public NeuralNetwork(int[] sizes_of_layers, Func<double, double> ActivationFunction, Func<double, double> ActivationFunctionDerivative, 
            Func<double, double> ActivationFunctionOutputLayers, Func<double, double> ActivationFunctionDerivativeOutputLayers, double learningrate)

        {
            this.ActivationFunctionDerivative = ActivationFunctionDerivative;
            this.ActivationFunctionDerivativeOutputLayers = ActivationFunctionDerivativeOutputLayers;
            this.learningrate = learningrate;
            number_of_hidden_layers = sizes_of_layers.Length - 2;
            for (int i = 0; i < number_of_hidden_layers; i++)
            {
                hiddenlayers.Add(new Layer(sizes_of_layers[i], sizes_of_layers[i + 1], ActivationFunction));
            }

            outputlayer = new Layer(sizes_of_layers[sizes_of_layers.Length - 2], sizes_of_layers[sizes_of_layers.Length - 1], ActivationFunctionOutputLayers);
        }

        /// <summary>
        /// Запускає алгоритм навчання неронної мережі
        /// </summary>
        /// <param name="datasetdirectory">шлях до директоріх з датасетом</param>
        /// <param name="numberOfEpochs">кількість епох</param>
        public void Train(string datasetdirectory, int numberOfEpochs)
            {
            string[] files = Directory.GetFiles(datasetdirectory, "*.png");

            for (int i = 0; i < numberOfEpochs; i++)
            {
                double[] errors = new double[outputlayer.number_of_neurons];
                double[] Feed_result = new double[10];
                string filePath = files[i];
                double[] targets = new double[10];
                for (int j = 0; j < 10; j++)
                {
                    targets[j] = 0;
                }

                int temp = (int)Char.GetNumericValue(filePath[18]);
                targets[temp] = 1;

                Feed_result = FeedForward(OpenImage(filePath));
                for (int j = 0; j < outputlayer.number_of_neurons; j++)
                {
                    errors[j] = targets[j] - Feed_result[j];
                }
                Backpropagation(errors);
            }

        }

        /// <summary>
        /// Запускає алгоритм зворотнього поширення помилки
        /// </summary>
        /// <param name="errors">відхилення від цілі</param>
        public void Backpropagation(double[] errors)
        {
            outputlayer.SetLocalGradient(errors, ActivationFunctionDerivativeOutputLayers);
            outputlayer.СhangeTheWeightsAndBias(learningrate);
            hiddenlayers[hiddenlayers.Count - 1].SetLocalGradient(outputlayer, ActivationFunctionDerivative);
            hiddenlayers[hiddenlayers.Count - 1].СhangeTheWeightsAndBias(learningrate);

            for (int i = hiddenlayers.Count - 2; i >= 0; i--)
            {
                hiddenlayers[i].SetLocalGradient(hiddenlayers[i + 1], ActivationFunctionDerivative);
                hiddenlayers[i].СhangeTheWeightsAndBias(learningrate);
            }
        }

        /// <summary>
        /// Визначає точність мережі
        /// </summary>
        /// <param name="datasetdirectory">шлях до директорії з тестовим датасетом</param>
        /// <param name="numberOfTest">кількість тестових прикладів</param>
        public void Test(string datasetdirectory, int numberOfTest)
        {
            string[] files = Directory.GetFiles(datasetdirectory, "*.png");

            double correct = 0;
            int incorrect = 0;

            for (int i = 0; i < numberOfTest; i++)
            {
                double[] feed_result = new double[10];

                string filePath = files[i];
                char target = filePath[17];
                int targetidx = (int)Char.GetNumericValue(target);

                feed_result = FeedForward(OpenImage(filePath));

                int idxmax = 0;
                double max = 0;

                for (int j = 0; j < 10; j++)
                {
                    if (max < feed_result[j])
                    {
                        max = feed_result[j];   
                        idxmax = j; 
                    }
                }
                if (targetidx == idxmax)
                {
                    correct++;
                }

                else incorrect++;
                
            }

            double percent_of_correct = correct / numberOfTest * 100.0;
       }

        /// <summary>
        /// Задає шару вхідний сигнал, встановлює індуковані поля неронів і повертає вихідний сигнал
        /// </summary>
        /// <param name="layer">шар</param>
        /// <param name="input">вхідний сигнал</param>
        /// <returns>выходной сигнал</returns>
        public double[] SetInducedLocalFieldAndReturnOutput(Layer layer, double[] input)
        {
            layer.input = input;
            double[] result = new double[layer.number_of_neurons];

            for (int i = 0; i < layer.number_of_neurons; i++)
            {
                layer.neurons[i].Feed(input);
            }

            for (int j = 0; j < layer.number_of_neurons; j++)
            {
                result[j] = layer.neurons[j].ActivationFunction(layer.neurons[j].InducedLocalField);
            }
            return result;
        }

        /// <summary>
        /// Запускає пряме поширення сигналу
        /// </summary>
        /// <param name="inputs">вхідний сигнал мережі</param>
        /// <returns></returns>
        public double[] FeedForward(double[] inputs)
        {
            for (int i = 0; i < number_of_hidden_layers; i++)
            {
                inputs = SetInducedLocalFieldAndReturnOutput(hiddenlayers[i], inputs);
            }
            inputs = SetInducedLocalFieldAndReturnOutput(outputlayer, inputs);
            return inputs;
        }

        /// <summary>
        /// Перетворює зображення на вхідний вектор чисел
        /// </summary>
        /// <param name="fileName">шлях до файлу</param>
        /// <returns></returns>
        public double[] OpenImage(string fileName)
        {
            Bitmap image = new Bitmap(fileName);
            int width = image.Width;
            int height = image.Height;

            double[] result = new double[width * height];
            int i = 0;

            for (int x = 0; x < height; x++)
            {
                for (int y = 0; y < width; y++)
                {
                    result[i] = (image.GetPixel(x, y).R) / 255.0;
                    i++;
                }
            }
            image.Dispose();

            return result;
        }
    }
}