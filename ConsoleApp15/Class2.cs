using Microsoft.ML.Data;

namespace ConsoleApp15
{
    class onceKomvec
    {
        [LoadColumn(0)]
        public float win { get; set; }

        [LoadColumn(1)]
        public float wout { get; set; }

        [LoadColumn(2)]
        public float kin { get; set; }

        [LoadColumn(0)]
        public float kout { get; set; }

        [LoadColumn(4)]
        [ColumnName("Features")]
        [VectorType(2)]
        public float Features { get; set; }
    }
}
