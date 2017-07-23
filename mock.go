package ann

// MockANNer is a mockable ANNer
type MockANNer struct {
	ANNFn func(q []float64, k int) []int
}

// ANN calls the underlying ANN method
func (nn *MockANNer) ANN(q []float64, k int) []int {
	return nn.ANNFn(q, k)
}
