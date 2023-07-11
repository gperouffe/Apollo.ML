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
            var modelPath = Path.Combine(".", "taxi-fare-model.zip");

            BuildTrainAndSave(mlContext, trainingSetPath, testingSetPath, modelPath);

            TestPrediction(mlContext, modelPath);
        }

        private static void BuildTrainAndSave(MLContext mlContext, string trainingSetPath, string testingSetPath, string modelPath)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<BikeDemandInput>(trainingSetPath, hasHeader: true, separatorChar: ',');
            var testDataView = mlContext.Data.LoadFromTextFile<BikeDemandInput>(testingSetPath, hasHeader: true, separatorChar: ',');

            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SeasonEncoded", inputColumnName: nameof(BikeDemandInput.Season))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "YearEncoded", inputColumnName: nameof(BikeDemandInput.Year)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "MonthEncoded", inputColumnName: nameof(BikeDemandInput.Month)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HourEncoded", inputColumnName: nameof(BikeDemandInput.Hour)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "HolidayEncoded", inputColumnName: nameof(BikeDemandInput.Holiday)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "WeekdayEncoded", inputColumnName: nameof(BikeDemandInput.Weekday)))
                            .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "WeatherEncoded", inputColumnName: nameof(BikeDemandInput.Weather)))
                            .Append(mlContext.Transforms.Concatenate("Features", "SeasonEncoded", "YearEncoded", "MonthEncoded", "HourEncoded", "HolidayEncoded", "WeekdayEncoded", "WeatherEncoded"
                            , nameof(BikeDemandInput.Temperature), nameof(BikeDemandInput.FeelingTemperature), nameof(BikeDemandInput.Humidity), nameof(BikeDemandInput.Windspeed)));

            var trainer = mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("=============== Apprentissage ===============");
            var model = trainingPipeline.Fit(trainingDataView);

            IDataView predictions = model.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("=============== Évaluation ===============");
            ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

            mlContext.Model.Save(model, trainingDataView.Schema, modelPath);
        }

        private static void TestPrediction(MLContext mlContext, string modelPath)
        {
            var input = new BikeDemandInput()
            {
                Season = 3,
                Year = 1,
                Month = 8,
                Hour = 10,
                Holiday = 0,
                Weekday = 4,
                WorkingDay = 1,
                Weather = 1,
                Temperature = 0.8f,
                FeelingTemperature = 0.7576f,
                Humidity = 0.55f,
                Windspeed = 0.2239f
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
