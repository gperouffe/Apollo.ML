using Microsoft.ML.Data;

namespace Apollo.ML.Examples.Example1.DataStructures
{
    public class ReviewPrediction
    {

        [ColumnName(@"Label")]
        public bool Label { get; set; }

        [ColumnName(@"Features")]
        public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public bool PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float Score { get; set; }

    }
}
