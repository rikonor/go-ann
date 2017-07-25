package ann

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func TestMRPTVsExhaustive(t *testing.T) {
	// Compare the accuracy of an MRPT-based ANNer vs a naive search
	n := 1000
	d := 10

	xs := randomMatrix(n, d)

	enn := NewExhaustiveNNer(xs)

	trees := 5
	depth := 5
	nn := NewMRPTANNer(trees, depth, xs)

	k := 1
	sameResultCount := 0

	expNum := 1000
	for i := 0; i < expNum; i++ {
		q := randomVector(d)

		indicesANN := nn.ANN(q, k)
		indicesENN := enn.ANN(q, k)

		if reflect.DeepEqual(indicesANN, indicesENN) {
			sameResultCount++
		}
	}

	// Calculate the hit ratio
	// Note: The hit ratio heavily depends on the tuning params of the MRPT algo
	hitRatio := float64(sameResultCount) / float64(expNum)
	fmt.Println(hitRatio)
}

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
