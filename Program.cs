using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace Lab1
{
    class FeedBackTrainingData
    {
        [ColumnName("Label"), LoadColumn(0)]
        public bool IsGood { get; set; }

        [ColumnName("FeedbackText"),LoadColumn(1)]
        public string FeedbackText { get; set; }

        [ColumnName("Features"), LoadColumn(2)]
        public float Features { get; set; }

    }


    class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }
    }

    class Program
    {
        static List<FeedBackTrainingData> trainingData = new List<FeedBackTrainingData>() { };
        static List<FeedBackTrainingData> testData = new List<FeedBackTrainingData>() { };

        static void LoadTestData()
        {
            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "good",
                    IsGood = true
                }
            );


            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "good work",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "awesome",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "awesome guys",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "decent",
                    IsGood = true
                }
            );


            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "decent work",
                    IsGood = true
                }
            );



            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "shitty!",
                    IsGood = false
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "is shitty!",
                    IsGood = false
                }
            );


            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "is horrible",
                    IsGood = false
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "is bad",
                    IsGood = false
                }
            );



            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "like",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "i like",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "i like it",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "nice",
                    IsGood = true
                }
            );


            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "love",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "we love it",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "superb",
                    IsGood = true
                }
            );

            testData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "superb work",
                    IsGood = true
                }
            );
        }

        static void LoadTraningData()
        {
            trainingData.Add(
                new FeedBackTrainingData() {
                    FeedbackText = "this is good",
                    IsGood = true
                }
            );

            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "this is horrible",
                    IsGood = false
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "this is bad",
                    IsGood = false
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "wtf this is shitty!",
                    IsGood = false
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "awesome",
                    IsGood = true
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "i like this it's really good",
                    IsGood = true
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "we love this it good",
                    IsGood = true
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "decent work",
                    IsGood = true
                }
            );


            trainingData.Add(
                new FeedBackTrainingData()
                {
                    FeedbackText = "superb work going on here guys",
                    IsGood = true
                }
            );
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Very first machine learning algorithm \nTo determine is a sentence is good or bad\n.");
            //step one load training data
            LoadTraningData();

            var mlContext = new MLContext();

            IDataView dataview = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(trainingData);

            var pipline = mlContext.Transforms
                .Text.FeaturizeText("Features", "FeedbackText")
                .Append(mlContext.BinaryClassification.Trainers
                .FastTree(numberOfLeaves: 50, numberOfTrees: 50, minimumExampleCountPerLeaf: 1));

            var model = pipline.Fit(dataview);

            LoadTestData();

            IDataView testdataview = mlContext.Data.LoadFromEnumerable<FeedBackTrainingData>(testData);

            var predictions = model.Transform(testdataview);

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            var predictionFunction = mlContext.Model.CreatePredictionEngine
                <FeedBackTrainingData, FeedBackPrediction>
                (model);

            while (true)
            {
                Console.WriteLine("Enter a feed back string: ");

                string feedBackString = Console.ReadLine().ToString();

                var feedbackInput = new FeedBackTrainingData();

                feedbackInput.FeedbackText = feedBackString;

                var feedbackPredicted = predictionFunction.Predict(feedbackInput);

                Console.WriteLine("Predicted Text :- " + feedbackPredicted.IsGood);

                Console.WriteLine("Accuracy of Prediction:- "+ metrics.Accuracy);
                Console.ReadLine();
            }
        }

    }
}
