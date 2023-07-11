using Apollo.ML.Examples.Common;
using Apollo.ML.Examples.Exercice2.DataStructures;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.utils;

namespace Apollo.ML.Examples.Exercice2
{
    public class BikeSharing
    {
        public static void Execute()
        {
            var mlContext = new MLContext();

            var trainingSetPath = Path.Combine(".", "Data", "bike-sharing-train.csv");
            var testingSetPath = Path.Combine(".", "Data", "bike-sharing-test.csv");
            var modelPath = Path.Combine(".", "bike-sharing-model.zip");

            BuildTrainAndSave(mlContext, trainingSetPath, testingSetPath, modelPath);

            TestPrediction(mlContext, modelPath);
        }

        private static void BuildTrainAndSave(MLContext mlContext, string trainingSetPath, string testingSetPath, string modelPath)
        {
            throw new NotImplementedException();
        }

        private static void TestPrediction(MLContext mlContext, string modelPath)
        {
            var input = new BikeDemandInput()
            {
            };

            ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<BikeDemandInput, BikeDemandPrediction>(trainedModel);

            var resultprediction = predEngine.Predict(input);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted bike user count : {resultprediction.PredictedCount:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
