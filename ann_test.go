package ann

import (
	"math/rand"
	"testing"
)

func BenchmarkMRPTANNer(b *testing.B) {
	n := 10000
	d := 100

	nn := NewMRPTANNer(3, 10, randomMatrix(n, d))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q := randomVector(d)
		b.StartTimer()

		nn.ANN(q, 1)
	}
}

func BenchmarkExhaustiveNNer(b *testing.B) {
	n := 10000
	d := 100

	nn := NewExhaustiveNNer(randomMatrix(n, d))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		q := randomVector(d)
		b.StartTimer()

		nn.ANN(q, 1)
	}
}

func randomMatrix(n, d int) [][]float64 {
	xs := [][]float64{}
	for i := 0; i < n; i++ {
		xs = append(xs, randomVector(d))
	}
	return xs
}

func randomVector(d int) []float64 {
	vs := []float64{}
	for j := 0; j < d; j++ {
		vs = append(vs, rand.NormFloat64())
	}
	return vs
}
