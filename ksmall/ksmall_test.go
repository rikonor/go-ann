package ksmall

import (
	"reflect"
	"testing"
)

func TestKSmallestIndices(t *testing.T) {
	vs := []float64{3.5, 5, 1.2}

	k := 2
	expectedIndices := []int{2, 0}

	indices := KSmallestIndices(vs, k)
	if !reflect.DeepEqual(indices, expectedIndices) {
		t.Fatalf("wrong indices: %v, expected: %v", indices, expectedIndices)
	}
}
