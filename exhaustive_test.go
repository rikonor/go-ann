package ann

import (
	"reflect"
	"testing"
)

func TestExhaustiveNNer(t *testing.T) {
	nn := NewExhaustiveNNer([][]float64{
		[]float64{0, 0, 0},
		[]float64{1.1, 1.1, 1.1},
		[]float64{2, 2, 2},
	})

	type testCase struct {
		q       []float64
		k       int
		indices []int
	}

	q := []float64{0.9, 0.9, 0.9}
	testCases := []testCase{
		testCase{q: q, k: 1, indices: []int{1}},
		testCase{q: q, k: 2, indices: []int{1, 0}},
		testCase{q: q, k: 3, indices: []int{1, 0, 2}},
	}

	for _, tc := range testCases {
		indices := nn.ANN(tc.q, tc.k)
		if !reflect.DeepEqual(indices, tc.indices) {
			t.Fatalf("incorrect indices: %v, expected %v", indices, tc.indices)
		}
	}
}

func TestExhaustiveNNerEmpty(t *testing.T) {
	nn := NewExhaustiveNNer([][]float64{})

	indices := nn.ANN([]float64{0, 0, 0}, 1)
	if len(indices) != 0 {
		t.Fatalf("too many results: %d, expected 0", len(indices))
	}
}
