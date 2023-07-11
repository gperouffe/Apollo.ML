using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Apollo.ML.Examples.Exercice2.DataStructures
{
    public class BikeDemandInput
    {
        [LoadColumn(2)]
        public float Season { get; set; }
        [LoadColumn(3)]
        public float Year { get; set; }
        [LoadColumn(4)]
        public float Month { get; set; }
        [LoadColumn(5)]
        public float Hour { get; set; }
        [LoadColumn(6)]
        public float Holiday { get; set; }
        [LoadColumn(7)]
        public float Weekday { get; set; }
        [LoadColumn(8)]
        public float WorkingDay { get; set; }
        [LoadColumn(9)]
        public float Weather { get; set; }
        [LoadColumn(10)]
        public float Temperature { get; set; }
        [LoadColumn(11)]
        public float FeelingTemperature { get; set; }
        [LoadColumn(12)]
        public float Humidity { get; set; }
        [LoadColumn(13)]
        public float Windspeed { get; set; }
        [LoadColumn(16)]
        [ColumnName("Label")]
        public float Count { get; set; }
    }
}
