using Microsoft.ML.Data;

namespace Apollo.ML.Examples.Exercice2.DataStructures
{
    public class BikeDemandInput
    {
        [LoadColumn(16)]
        [ColumnName("Label")]
        public float Count { get; set; }
    }
}
