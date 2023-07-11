using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Apollo.ML.Examples.Exercice1.DataStructures
{
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction;

        public float Probability;

        public float Score;
    }
}
