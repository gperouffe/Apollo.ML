using Apollo.ML.Examples.Example1.DataStructures;
using Microsoft.ML;
using System;
using System.Diagnostics;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;

namespace Apollo.ML.Examples.Example1
{
    /// <summary>
    /// Sentiment Analysis demo
    /// </summary>
    public static class SentimentAnalysis
    {
        public static void Execute()
        {

            // Création du context
            var mlContext = new MLContext();

            TrainTestData split = LoadData(mlContext);

            ITransformer newModel = PrepareDataAndTrain(mlContext, split.TrainSet);

            EvaluateModel(mlContext, split.TestSet, newModel);

            PromptUserAndPredict(mlContext, newModel);
        }


        /// <summary>
        /// Chargement des données et séparation en jeux train/test
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        private static TrainTestData LoadData(MLContext mlContext)
        {
            Console.WriteLine("Chargement des données...");
            Stopwatch stopWatch = Stopwatch.StartNew();

            var datasetPath = Path.Combine(".", "Data", "allocine-train.tsv");

            var data = mlContext.Data.LoadFromTextFile<ReviewInput>(datasetPath, allowQuoting: true, separatorChar: '\t');

            var split = mlContext.Data.TrainTestSplit(data);

            stopWatch.Stop(); 
            Console.WriteLine("Terminé en {0} s", stopWatch.ElapsedMilliseconds / 1000f);
            return split;
        }

        /// <summary>
        /// Préparation des données et entrainement du modèle
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="split"></param>
        /// <returns></returns>
        private static ITransformer PrepareDataAndTrain(MLContext mlContext, IDataView trainData)
        {
            Console.WriteLine("Préparation des données et entrainement du modèle...");
            Stopwatch stopWatch = Stopwatch.StartNew();

            var pipeline = mlContext.Transforms.Text.FeaturizeText(inputColumnName: @"Review", outputColumnName: @"Features")
                                    .Append(mlContext.Transforms.NormalizeMinMax(@"Features"))
                                    .Append(mlContext.Transforms.CopyColumns(@"Label", @"Sentiment"))
                                    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());
            var newModel = pipeline.Fit(trainData);

            stopWatch.Stop();
            Console.WriteLine("Terminé en {0} s", stopWatch.ElapsedMilliseconds / 1000f);

            return newModel;
        }

        /// <summary>
        /// Évaluation du modèle
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="split"></param>
        /// <param name="newModel"></param>
        private static void EvaluateModel(MLContext mlContext, IDataView testData, ITransformer model)
        {
            Console.WriteLine("Évaluation du modèle...");
            Stopwatch stopWatch = Stopwatch.StartNew();
            
            var testDataPredictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(testDataPredictions);
            Console.WriteLine($"Précision: {metrics.Accuracy}");

            stopWatch.Stop();
            Console.WriteLine("Terminé en {0} s", stopWatch.ElapsedMilliseconds / 1000f);
        }

        /// <summary>
        /// Prompts the user for a review and outputs the prediction of the model
        /// </summary>
        /// <param name="mlContext"></param>
        /// <param name="newModel"></param>
        private static void PromptUserAndPredict(MLContext mlContext, ITransformer newModel)
        {

            // Création de l'engine de prédiction
            var engine = mlContext.Model.CreatePredictionEngine<ReviewInput, ReviewPrediction>(newModel);

            string? prompt = null;
            while (prompt != "stop")
            {
                Console.WriteLine("Enter your review (or 'stop') : ");
                prompt = Console.ReadLine() ?? "";

                // Prédiction
                //var prediction = SentimentAnalysis.Predict(new() { Col0 = prompt });
                var prediction = engine.Predict(new() { Review = prompt });

                if (prediction.PredictedLabel)
                {
                    Console.WriteLine($"Review is positive. (Score = {prediction.Score})");
                }
                else
                {
                    Console.WriteLine($"Review is negative. (Score = {prediction.Score})");
                }
                Console.WriteLine();
            }
        }
    }
}
