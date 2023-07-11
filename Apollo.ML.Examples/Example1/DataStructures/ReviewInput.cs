using Microsoft.ML.Data;

namespace Apollo.ML.Examples.Example1.DataStructures
{
    public class ReviewInput
    {
        [LoadColumn(0)]
        public string Review { get; set; }

        [LoadColumn(1)]
        public bool Sentiment { get; set; }

    }
}
