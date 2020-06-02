using Microsoft.ML.Data;

namespace ConsoleApp15
{
    class onceKom
    {
        [LoadColumn(0)]
        public float win { get; set; }

        [LoadColumn(1)]
        public float wout { get; set; }

        [LoadColumn(2)]
        public float kin { get; set; }

        [LoadColumn(0)]
        public float kout { get; set; }

        [LoadColumn(4,5), VectorType(2)]
        [ColumnName("Features")]
        public float[] Features { get; set; }

        [LoadColumn(6)]
        public bool Label { get; set; }
    }
}
