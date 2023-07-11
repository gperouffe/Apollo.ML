using Apollo.ML.Examples.Common;
using Apollo.ML.Examples.Exercice1.DataStructures;
using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace Apollo.ML.Examples.Exercice1
{
    public class HeartDisease
    {
        public static void Execute()
        {
            var mlContext = new MLContext();

            var trainingSetPath = Path.Combine(".", "Data", "heart-disease-train.csv");
            var testingSetPath = Path.Combine(".", "Data", "heart-disease-test.csv");
            var modelPath = Path.Combine(".", "taxi-fare-model.zip");
            BuildTrainAndSave(mlContext, trainingSetPath, testingSetPath, modelPath);

            TestPrediction(mlContext, modelPath);
        }

        private static void BuildTrainAndSave(MLContext mlContext, string trainingSetPath, string testingSetPath, string modelPath)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<HeartDataInput>(trainingSetPath, separatorChar: ';');
            var testDataView = mlContext.Data.LoadFromTextFile<HeartDataInput>(testingSetPath, separatorChar: ';');

            var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));

            ITransformer trainedModel = pipeline.Fit(trainingDataView);

            var predictions = trainedModel.Transform(testDataView);

            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics("FastTree", metrics);

            mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
        }

        private static void TestPrediction(MLContext mlContext, string modelPath)
        {
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            // Create prediction engine related to the loaded trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HeartDataInput, HeartPrediction>(trainedModel);

            foreach (var heartData in HeartSampleData.heartDataList)
            {
                var prediction = predictionEngine.Predict(heartData);

                Console.WriteLine($"=============== Single Prediction  ===============");
                Console.WriteLine($"Age: {heartData.Age} ");
                Console.WriteLine($"Sex: {heartData.Sex} ");
                Console.WriteLine($"Cp: {heartData.Cp} ");
                Console.WriteLine($"TrestBps: {heartData.TrestBps} ");
                Console.WriteLine($"Chol: {heartData.Chol} ");
                Console.WriteLine($"Fbs: {heartData.Fbs} ");
                Console.WriteLine($"RestEcg: {heartData.RestEcg} ");
                Console.WriteLine($"Thalac: {heartData.Thalac} ");
                Console.WriteLine($"Exang: {heartData.Exang} ");
                Console.WriteLine($"OldPeak: {heartData.OldPeak} ");
                Console.WriteLine($"Slope: {heartData.Slope} ");
                Console.WriteLine($"Ca: {heartData.Ca} ");
                Console.WriteLine($"Thal: {heartData.Thal} ");
                Console.WriteLine($"Prediction Value: {prediction.Prediction} ");
                Console.WriteLine($"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease")} ");
                Console.WriteLine($"Probability: {prediction.Probability} ");
                Console.WriteLine($"==================================================");
                Console.WriteLine("");
                Console.WriteLine("");
            }

        }
    }
}
