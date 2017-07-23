package ann

import (
	"reflect"
	"testing"
)

func TestMappedANNer(t *testing.T) {
	nn := &MockANNer{
		ANNFn: func(q []float64, k int) []int {
			// Return a fake series of indices
			return []int{3, 0}
		},
	}

	// Wrap the mock with a MappedANNer
	mnn := NewMappedANNer(nn, []string{"1", "2", "3", "4"})

	// Call the mnn with some fake query point and value k (they get ignored anyway)
	vs := mnn.ANN([]float64{1, 2, 3}, 2)

	if !reflect.DeepEqual(vs, []string{"4", "1"}) {
		t.Fatalf("incorrect values: %v", vs)
	}
}
