package ksmall

import "sort"

// KSmallestIndices takes a series of values and an integer k
// and returns the indices of the smallest values
func KSmallestIndices(vs []float64, k int) []int {
	indexedValues := wrapIndexedValues(vs)

	sort.Sort(byIndexedValue(indexedValues))

	// Take k smallest indexes
	indices := []int{}
	for i := 0; i < k; i++ {
		indices = append(indices, indexedValues[i].index)
	}

	return indices
}

type indexedValue struct {
	index int
	value float64
}

func wrapIndexedValues(vs []float64) []indexedValue {
	indexedValues := []indexedValue{}
	for i, d := range vs {
		indexedValues = append(indexedValues, indexedValue{
			index: i,
			value: d,
		})
	}
	return indexedValues
}

type byIndexedValue []indexedValue

func (a byIndexedValue) Len() int           { return len(a) }
func (a byIndexedValue) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byIndexedValue) Less(i, j int) bool { return a[i].value < a[j].value }
