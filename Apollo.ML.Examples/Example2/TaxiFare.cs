using Apollo.ML.Examples.Common;
using Apollo.ML.Examples.Example2.DataStructures;
using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace Apollo.ML.Examples.Example2
{
    public class TaxiFare
    {
        public static void Execute()
        {
            var mlContext = new MLContext();

            var trainingSetPath = Path.Combine(".", "Data", "taxi-fare-train.csv");
            var testingSetPath = Path.Combine(".", "Data", "taxi-fare-test.csv");
            var modelPath = Path.Combine(".", "taxi-fare-model.zip");
           BuildTrainAndSave(mlContext, trainingSetPath, testingSetPath, modelPath);

            TestSinglePrediction(mlContext, modelPath);
        }

        private static void BuildTrainAndSave(MLContext mlContext, string trainingSetPath, string testingSetPath, string modelPath)
        {
            Console.WriteLine("=============== Chargement des données ===============");
            var trainingDV = mlContext.Data.LoadFromTextFile<TaxiFareInput>(trainingSetPath, separatorChar: ',');
            var testingDV = mlContext.Data.LoadFromTextFile<TaxiFareInput>(testingSetPath, separatorChar: ',');

            var dataProcessPipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TaxiFareInput.FareAmount))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(TaxiFareInput.VendorId)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(TaxiFareInput.RateCode)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(TaxiFareInput.PaymentType)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiFareInput.PassengerCount)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiFareInput.TripTime)))
                            .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiFareInput.TripDistance)))
                            .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(TaxiFareInput.PassengerCount)
                            , nameof(TaxiFareInput.TripTime), nameof(TaxiFareInput.TripDistance)));

            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("=============== Apprentissage ===============");
            var model = trainingPipeline.Fit(trainingDV);

            IDataView predictions = model.Transform(testingDV);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("=============== Évaluation ===============");
            ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

            mlContext.Model.Save(model, trainingDV.Schema, modelPath);
        }

        private static void TestSinglePrediction(MLContext mlContext, string modelPath)
        {

            var taxiTripSample = new TaxiFareInput()
            {
                //VTS,1,1,1380,4.91,CRD,19.0
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1380,
                TripDistance = 4.91f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            ITransformer trainedModel = mlContext.Model.Load(modelPath, out var modelInputSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<TaxiFareInput, TaxiTripFareOutput>(trainedModel);

            var resultprediction = predEngine.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
