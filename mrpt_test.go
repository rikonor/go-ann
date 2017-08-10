package ann

import (
	"reflect"
	"testing"
)

func TestMRPTNNer(t *testing.T) {
	nn := NewMRPTANNer(1, 3, [][]float64{
		[]float64{0, 0, 0, 0},
		[]float64{1.1, 1.1, 1.1, 1.1},
		[]float64{2, 2, 2, 2},
		[]float64{-1, 1, 2, 2},
		[]float64{2, 2, -1, 2},
		[]float64{2, 3, 2, -4},
		[]float64{-2, -3, 2, -4},
		[]float64{-2, -3, -2, 4},
	})

	q := []float64{1, 1, 1, 1}
	indices := nn.ANN(q, 1)

	// Should be closest to vector #1
	expectedIndices := []int{1}

	if !reflect.DeepEqual(indices, expectedIndices) {
		t.Fatalf("expected nn to be %v, got indices %v", expectedIndices, indices)
	}
}

func TestMedian(t *testing.T) {
	type testCase struct {
		vs []float64
		m  float64
	}

	testCases := []testCase{
		testCase{vs: []float64{1}, m: 1},
		testCase{vs: []float64{1, 2, 3}, m: 2},
		testCase{vs: []float64{1, 2, 3, 4}, m: 2.5},
	}

	for _, tc := range testCases {
		m := median(tc.vs)
		if m != tc.m {
			t.Fatalf("wrong median: %f, expected %f", m, tc.m)
		}
	}
}
