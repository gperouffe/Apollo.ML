using Microsoft.ML.Data;

namespace Apollo.ML.Examples.Example2.DataStructures
{
    public class TaxiTripFareOutput
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}
