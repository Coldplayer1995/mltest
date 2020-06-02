using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ConsoleApp15
{
    class Program
    {
        static void Main(string[] args)
        {

            MLContext mLContext = new MLContext();

            IDataView dane = mLContext.Data.LoadFromTextFile<onceKom>("C:/Users/Patryk/source/repos/ConsoleApp15/Zeszyt1.csv", separatorChar:',',hasHeader: true);

            var split = mLContext.Data
                .TrainTestSplit(dane, testFraction: 0.2);

            var trainSet = mLContext.Data
               .CreateEnumerable<onceKom>(split.TrainSet, reuseRowObject: false);

            var testSet = mLContext.Data
                .CreateEnumerable<onceKom>(split.TestSet, reuseRowObject: false);

            var trening = mLContext.Data.LoadFromEnumerable(trainSet);
            var test = mLContext.Data.LoadFromEnumerable(testSet);

            PrintPreviewRows(trainSet, testSet);
            //Console.ReadLine();
            var ustawieniaSvm = new LinearSvmTrainer.Options
            {
                BatchSize = 10,
                PerformProjection = true,
                NumberOfIterations = 10
            };

            var pipeline = mLContext.BinaryClassification.Trainers.LinearSvm(ustawieniaSvm);

            var model = pipeline.Fit(trening);

            var transformedTestData = model.Transform(test);
            var predictions = mLContext.Data
               .CreateEnumerable<onceKom>(transformedTestData,
               reuseRowObject: false).ToList();

            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.win}, "
                    + $"Prediction: {p.Features}");

            Console.ReadLine();
        }

        private static void PrintPreviewRows(IEnumerable<onceKom> trainSet, IEnumerable<onceKom> testSet)//dla testu czy działa podział danych
        {
            Console.WriteLine($"Zbiór treinigowy.");
            foreach (var row in trainSet)
                Console.WriteLine($"{row.kin}, {row.kout}");

            Console.WriteLine($"\nZbiór testowy.");
            foreach (var row in testSet)
                Console.WriteLine($"{row.kin}, {row.kout}");
        }
    }
}
