using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Apollo.ML.Examples.Exercice2.DataStructures
{
    public class BikeDemandPrediction
    {
        [ColumnName("Score")]
        public float PredictedCount;
        public float[] Features { get; set; }
        public float[] FeatureContributions { get; set; }
    }
}
