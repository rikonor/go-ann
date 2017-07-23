package ann

// MappedANNer is an ANNer that will return the string values
// associated with the k nearest neighbors instead of their indices
type MappedANNer interface {
	ANN(q []float64, k int) []string
}

// MockMappedANNer is a mockable MappedANNer
type MockMappedANNer struct {
	ANNFn func(q []float64, k int) []string
}

// ANN calls the underlying ANN method
func (nn *MockMappedANNer) ANN(q []float64, k int) []string {
	return nn.ANNFn(q, k)
}

// NewMappedANNer creates a new MappedANNer given an existing ANNer
// and a mapping from indices to a series of string values
func NewMappedANNer(nn ANNer, mapping []string) MappedANNer {
	return &MockMappedANNer{
		ANNFn: func(q []float64, k int) []string {
			// Perform initial search retrieving indices
			indices := nn.ANN(q, k)

			// Retrieve the series of associated string values
			vs := []string{}
			for _, i := range indices {
				vs = append(vs, mapping[i])
			}

			return vs
		},
	}
}
